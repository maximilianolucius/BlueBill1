#!/usr/bin/env python3
"""
Rebuild FAISS indexes from intact embeddings in SQLite database
"""

import sqlite3
import pickle
import base64
import numpy as np
import faiss
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_PATH = 'smartdoc_persistence.db'

def deserialize_numpy_array(data):
    """Deserialize numpy array from string"""
    try:
        return pickle.loads(base64.b64decode(data.encode('utf-8')))
    except Exception as e:
        logger.error(f"Error deserializing numpy array: {e}")
        return np.array([])

def serialize_faiss_index(index):
    """Serialize FAISS index to base64 string"""
    return base64.b64encode(faiss.serialize_index(index)).decode('utf-8')

def rebuild_all_faiss_indexes():
    """Rebuild all FAISS indexes from embeddings"""
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        # Get all documents with embeddings
        cursor.execute('SELECT doc_id, embeddings FROM documents ORDER BY doc_id')
        documents = cursor.fetchall()
        
        logger.info(f"Found {len(documents)} documents to rebuild FAISS indexes for")
        
        # Clear existing broken FAISS indexes
        cursor.execute('DELETE FROM faiss_indexes')
        cursor.execute('DELETE FROM global_faiss_index')
        
        # Rebuild individual document indexes
        rebuilt_count = 0
        all_embeddings = []
        global_metadata = []
        
        for doc_id, embeddings_data in documents:
            try:
                # Deserialize embeddings
                embeddings = deserialize_numpy_array(embeddings_data)
                
                if embeddings.size == 0:
                    logger.warning(f"Skipping doc {doc_id} - no valid embeddings")
                    continue
                
                # Ensure proper dtype
                embeddings = embeddings.astype('float32')
                
                # Create FAISS index
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)
                
                # Serialize and save
                index_blob = serialize_faiss_index(index)
                
                cursor.execute('''
                    INSERT INTO faiss_indexes (doc_id, faiss_index, dimension)
                    VALUES (?, ?, ?)
                ''', (doc_id, index_blob, dimension))
                
                # Collect for global index
                all_embeddings.extend(embeddings)
                
                # Get document metadata for global index
                cursor.execute('SELECT filename, chunks FROM documents WHERE doc_id = ?', (doc_id,))
                doc_row = cursor.fetchone()
                if doc_row:
                    filename = doc_row[0]
                    chunks_json = doc_row[1]
                    import json
                    chunks = json.loads(chunks_json)
                    
                    for i, chunk in enumerate(chunks):
                        global_metadata.append({
                            'doc_id': doc_id,
                            'filename': filename,
                            'chunk_index': i,
                            'chunk_text': chunk,
                            'global_index': len(global_metadata)
                        })
                
                rebuilt_count += 1
                logger.info(f"✅ Rebuilt FAISS index for doc {doc_id} ({embeddings.shape[0]} vectors)")
                
            except Exception as e:
                logger.error(f"❌ Failed to rebuild FAISS index for doc {doc_id}: {e}")
        
        # Build global FAISS index
        if all_embeddings:
            try:
                global_embeddings = np.array(all_embeddings, dtype='float32')
                global_index = faiss.IndexFlatL2(global_embeddings.shape[1])
                global_index.add(global_embeddings)
                
                # Save global index
                global_index_blob = serialize_faiss_index(global_index)
                cursor.execute('''
                    INSERT INTO global_faiss_index (id, faiss_index, dimension, total_vectors)
                    VALUES (1, ?, ?, ?)
                ''', (global_index_blob, global_embeddings.shape[1], global_embeddings.shape[0]))
                
                # Save global metadata
                cursor.execute('DELETE FROM global_chunk_metadata')
                for meta in global_metadata:
                    cursor.execute('''
                        INSERT INTO global_chunk_metadata 
                        (doc_id, filename, chunk_index, chunk_text, global_index)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        meta['doc_id'], meta['filename'], meta['chunk_index'], 
                        meta['chunk_text'], meta['global_index']
                    ))
                
                logger.info(f"✅ Built global FAISS index with {global_embeddings.shape[0]} vectors")
                
            except Exception as e:
                logger.error(f"❌ Failed to build global FAISS index: {e}")
        
        conn.commit()
        logger.info(f"🎉 Successfully rebuilt {rebuilt_count} FAISS indexes")
        
        # Test one rebuilt index
        cursor.execute('SELECT faiss_index FROM faiss_indexes LIMIT 1')
        test_row = cursor.fetchone()
        if test_row:
            try:
                test_index = faiss.deserialize_index(base64.b64decode(test_row[0].encode('utf-8')))
                logger.info(f"✅ Test deserialization successful: {test_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"❌ Test deserialization failed: {e}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    print(f"🔄 Starting FAISS index rebuild at {datetime.now()}")
    rebuild_all_faiss_indexes()
    print(f"✅ FAISS index rebuild completed at {datetime.now()}")
