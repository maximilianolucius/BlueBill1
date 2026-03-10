#!/usr/bin/env python3
"""
Fix FAISS serialization format in database
"""

import sqlite3
import pickle
import base64
import numpy as np
import faiss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_PATH = 'smartdoc_persistence.db'

def fix_faiss_serialization():
    """Fix FAISS serialization format issues"""
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Test current serialization
        cursor.execute('SELECT COUNT(*) FROM faiss_indexes')
        count = cursor.fetchone()[0]
        logger.info(f"Found {count} FAISS indexes to check")
        
        # Check one index
        cursor.execute('SELECT doc_id, faiss_index FROM faiss_indexes LIMIT 1')
        row = cursor.fetchone()
        if row:
            doc_id, index_data = row
            logger.info(f"Testing index for doc {doc_id}")
            logger.info(f"Data type: {type(index_data)}")
            
            # Try different deserialization approaches
            try:
                if isinstance(index_data, str):
                    # Method 1: Direct base64 decode
                    logger.info("Trying base64 decode from string...")
                    decoded = base64.b64decode(index_data.encode('utf-8'))
                    index = faiss.deserialize_index(decoded)
                    logger.info(f"✅ Method 1 works: {index.ntotal} vectors")
                    return True
                    
            except Exception as e1:
                logger.error(f"Method 1 failed: {e1}")
                
                try:
                    if isinstance(index_data, str):
                        # Method 2: Try direct string as bytes
                        logger.info("Trying direct string as bytes...")
                        index = faiss.deserialize_index(index_data.encode('utf-8'))
                        logger.info(f"✅ Method 2 works: {index.ntotal} vectors")
                        return True
                        
                except Exception as e2:
                    logger.error(f"Method 2 failed: {e2}")
                    
                    try:
                        # Method 3: Re-serialize with pickle+base64
                        logger.info("Trying pickle+base64 approach...")
                        
                        # Get original embeddings and rebuild
                        cursor.execute('SELECT embeddings FROM documents WHERE doc_id = ?', (doc_id,))
                        emb_row = cursor.fetchone()
                        if emb_row:
                            embeddings_data = emb_row[0]
                            embeddings = pickle.loads(base64.b64decode(embeddings_data.encode('utf-8')))
                            
                            # Create fresh index
                            fresh_index = faiss.IndexFlatL2(embeddings.shape[1])
                            fresh_index.add(embeddings.astype('float32'))
                            
                            # Use different serialization: pickle + base64
                            serialized = pickle.dumps(faiss.serialize_index(fresh_index))
                            serialized_b64 = base64.b64encode(serialized).decode('utf-8')
                            
                            # Test deserialization
                            test_deserial = pickle.loads(base64.b64decode(serialized_b64.encode('utf-8')))
                            test_index = faiss.deserialize_index(test_deserial)
                            logger.info(f"✅ Method 3 works: {test_index.ntotal} vectors")
                            
                            # Update database with new format
                            logger.info("Updating all indexes with new serialization format...")
                            update_all_indexes_new_format()
                            return True
                            
                    except Exception as e3:
                        logger.error(f"Method 3 failed: {e3}")
        
        return False
        
    finally:
        conn.close()

def update_all_indexes_new_format():
    """Update all FAISS indexes with pickle+base64 format"""
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Get all documents
        cursor.execute('SELECT doc_id, embeddings FROM documents')
        documents = cursor.fetchall()
        
        logger.info(f"Updating {len(documents)} FAISS indexes...")
        
        for doc_id, embeddings_data in documents:
            try:
                # Deserialize embeddings
                embeddings = pickle.loads(base64.b64decode(embeddings_data.encode('utf-8')))
                embeddings = embeddings.astype('float32')
                
                # Create FAISS index
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                
                # New serialization format: pickle(faiss_bytes) + base64
                faiss_bytes = faiss.serialize_index(index)
                pickled = pickle.dumps(faiss_bytes)
                serialized = base64.b64encode(pickled).decode('utf-8')
                
                # Update database
                cursor.execute('''
                    UPDATE faiss_indexes 
                    SET faiss_index = ? 
                    WHERE doc_id = ?
                ''', (serialized, doc_id))
                
                logger.info(f"✅ Updated FAISS index for doc {doc_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to update doc {doc_id}: {e}")
        
        # Update global index too
        cursor.execute('SELECT embeddings FROM documents')
        all_docs = cursor.fetchall()
        
        all_embeddings = []
        for doc_data in all_docs:
            embeddings = pickle.loads(base64.b64decode(doc_data[0].encode('utf-8')))
            all_embeddings.append(embeddings)
        
        if all_embeddings:
            global_embeddings = np.vstack(all_embeddings).astype('float32')
            global_index = faiss.IndexFlatL2(global_embeddings.shape[1])
            global_index.add(global_embeddings)
            
            # Same new format for global index
            global_faiss_bytes = faiss.serialize_index(global_index)
            global_pickled = pickle.dumps(global_faiss_bytes)
            global_serialized = base64.b64encode(global_pickled).decode('utf-8')
            
            cursor.execute('''
                UPDATE global_faiss_index 
                SET faiss_index = ? 
                WHERE id = 1
            ''', (global_serialized,))
            
            logger.info(f"✅ Updated global FAISS index")
        
        conn.commit()
        logger.info("🎉 All indexes updated with new format")
        
        # Test the new format
        cursor.execute('SELECT faiss_index FROM faiss_indexes LIMIT 1')
        test_row = cursor.fetchone()
        if test_row:
            test_data = test_row[0]
            test_unpickled = pickle.loads(base64.b64decode(test_data.encode('utf-8')))
            test_index = faiss.deserialize_index(test_unpickled)
            logger.info(f"✅ New format test successful: {test_index.ntotal} vectors")
        
    finally:
        conn.close()

if __name__ == "__main__":
    logger.info("🔧 Starting FAISS serialization fix...")
    success = fix_faiss_serialization()
    if success:
        logger.info("✅ FAISS serialization fix completed successfully")
    else:
        logger.error("❌ FAISS serialization fix failed")
