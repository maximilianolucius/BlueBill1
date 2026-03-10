#!/usr/bin/env python3
"""
Job Statistics Query Tool
==========================

Script para consultar estadísticas de jobs de clasificación fiscal desde la base de datos SQLite.

Uso:
    python job_stats.py                    # Ver estadísticas generales
    python job_stats.py --status pending   # Ver solo jobs pendientes
    python job_stats.py --recent 10        # Ver últimos 10 jobs
    python job_stats.py --export stats.json # Exportar estadísticas a JSON
"""

import sqlite3
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import argparse


# Configuración
DEFAULT_DB_PATH = os.getenv('SMARTDOC_DB_PATH', 'smartdoc_persistence.db')
JOBS_TABLE = 'classification_jobs'


class JobStatsAnalyzer:
    """Analizador de estadísticas de jobs"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._check_database()

    def _check_database(self):
        """Verificar que la base de datos existe"""
        if not Path(self.db_path).exists():
            print(f"❌ Error: Database not found at {self.db_path}")
            sys.exit(1)

    def _get_connection(self) -> sqlite3.Connection:
        """Obtener conexión a la base de datos"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_total_stats(self) -> Dict:
        """Obtener estadísticas generales"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Total de jobs
        cursor.execute(f"SELECT COUNT(*) as total FROM {JOBS_TABLE}")
        total = cursor.fetchone()['total']

        # Jobs por estado
        cursor.execute(f"""
            SELECT status, COUNT(*) as count
            FROM {JOBS_TABLE}
            GROUP BY status
        """)
        by_status = {row['status']: row['count'] for row in cursor.fetchall()}

        # Duración promedio (solo completados)
        cursor.execute(f"""
            SELECT
                AVG(actual_duration) as avg_duration,
                MIN(actual_duration) as min_duration,
                MAX(actual_duration) as max_duration
            FROM {JOBS_TABLE}
            WHERE status = 'completed' AND actual_duration IS NOT NULL
        """)
        duration = cursor.fetchone()

        # Tasa de éxito
        completed = by_status.get('completed', 0)
        failed = by_status.get('failed', 0)
        total_finished = completed + failed
        success_rate = (completed / total_finished * 100) if total_finished > 0 else 0

        # Jobs de las últimas 24 horas
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute(f"""
            SELECT COUNT(*) as recent
            FROM {JOBS_TABLE}
            WHERE created_at >= ?
        """, (yesterday,))
        recent_24h = cursor.fetchone()['recent']

        conn.close()

        return {
            'total_jobs': total,
            'by_status': by_status,
            'success_rate': round(success_rate, 2),
            'duration_stats': {
                'avg_seconds': round(duration['avg_duration'], 2) if duration['avg_duration'] else None,
                'min_seconds': duration['min_duration'],
                'max_seconds': duration['max_duration']
            },
            'recent_24h': recent_24h
        }

    def get_jobs_by_status(self, status: str) -> List[Dict]:
        """Obtener jobs filtrados por estado"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT job_id, status, created_at, started_at, completed_at,
                   progress, current_step, error_message, actual_duration
            FROM {JOBS_TABLE}
            WHERE status = ?
            ORDER BY created_at DESC
        """, (status,))

        jobs = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return jobs

    def get_recent_jobs(self, limit: int = 10) -> List[Dict]:
        """Obtener los N jobs más recientes"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT job_id, status, created_at, started_at, completed_at,
                   progress, current_step, error_message, actual_duration
            FROM {JOBS_TABLE}
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        jobs = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return jobs

    def get_error_analysis(self) -> Dict:
        """Análisis de errores"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Errores más comunes
        cursor.execute(f"""
            SELECT error_message, COUNT(*) as count
            FROM {JOBS_TABLE}
            WHERE status = 'failed' AND error_message IS NOT NULL
            GROUP BY error_message
            ORDER BY count DESC
            LIMIT 10
        """)

        common_errors = [
            {'error': row['error_message'], 'count': row['count']}
            for row in cursor.fetchall()
        ]

        # Total de fallos
        cursor.execute(f"""
            SELECT COUNT(*) as total_failures
            FROM {JOBS_TABLE}
            WHERE status = 'failed'
        """)
        total_failures = cursor.fetchone()['total_failures']

        conn.close()

        return {
            'total_failures': total_failures,
            'common_errors': common_errors
        }

    def get_time_distribution(self) -> Dict:
        """Distribución temporal de jobs"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Jobs por hora del día
        cursor.execute(f"""
            SELECT
                strftime('%H', created_at) as hour,
                COUNT(*) as count
            FROM {JOBS_TABLE}
            GROUP BY hour
            ORDER BY hour
        """)
        by_hour = {row['hour']: row['count'] for row in cursor.fetchall()}

        # Jobs por día de la semana
        cursor.execute(f"""
            SELECT
                strftime('%w', created_at) as dow,
                COUNT(*) as count
            FROM {JOBS_TABLE}
            GROUP BY dow
            ORDER BY dow
        """)

        days_map = {
            '0': 'Sunday', '1': 'Monday', '2': 'Tuesday',
            '3': 'Wednesday', '4': 'Thursday', '5': 'Friday', '6': 'Saturday'
        }
        by_day = {days_map[row['dow']]: row['count'] for row in cursor.fetchall()}

        conn.close()

        return {
            'by_hour': by_hour,
            'by_day_of_week': by_day
        }

    def get_job_details(self, job_id: str) -> Optional[Dict]:
        """Obtener detalles completos de un job específico"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT *
            FROM {JOBS_TABLE}
            WHERE job_id = ?
        """, (job_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            job = dict(row)
            # Deserializar JSON fields
            if job['factura_data']:
                job['factura_data'] = json.loads(job['factura_data'])
            if job['result_data']:
                job['result_data'] = json.loads(job['result_data'])
            return job
        return None

    def cleanup_old_jobs(self, days: int = 30, dry_run: bool = True) -> Dict:
        """Limpiar jobs antiguos (por defecto en modo dry-run)"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        # Contar jobs a eliminar
        cursor.execute(f"""
            SELECT COUNT(*) as count
            FROM {JOBS_TABLE}
            WHERE created_at < ? AND status IN ('completed', 'failed')
        """, (cutoff_date,))

        count = cursor.fetchone()['count']

        if not dry_run and count > 0:
            cursor.execute(f"""
                DELETE FROM {JOBS_TABLE}
                WHERE created_at < ? AND status IN ('completed', 'failed')
            """, (cutoff_date,))
            conn.commit()

        conn.close()

        return {
            'jobs_to_delete': count,
            'cutoff_date': cutoff_date,
            'dry_run': dry_run
        }


def print_stats_summary(stats: Dict):
    """Imprimir resumen de estadísticas"""
    print("\n" + "="*70)
    print(" 📊 JOB STATISTICS SUMMARY")
    print("="*70)

    print(f"\n📈 GENERAL")
    print(f"   Total Jobs:     {stats['total_jobs']}")
    print(f"   Last 24h:       {stats['recent_24h']}")
    print(f"   Success Rate:   {stats['success_rate']}%")

    print(f"\n📋 BY STATUS")
    for status, count in stats['by_status'].items():
        emoji = {
            'pending': '⏳',
            'processing': '⚙️',
            'completed': '✅',
            'failed': '❌'
        }.get(status, '❓')
        print(f"   {emoji} {status:12} {count}")

    if stats['duration_stats']['avg_seconds']:
        print(f"\n⏱️  DURATION (completed jobs)")
        print(f"   Average:   {stats['duration_stats']['avg_seconds']}s")
        print(f"   Min:       {stats['duration_stats']['min_seconds']}s")
        print(f"   Max:       {stats['duration_stats']['max_seconds']}s")

    print("\n" + "="*70 + "\n")


def print_jobs_table(jobs: List[Dict], title: str = "Jobs"):
    """Imprimir tabla de jobs"""
    if not jobs:
        print(f"\n❌ No {title.lower()} found.\n")
        return

    print(f"\n{'='*130}")
    print(f" {title}")
    print('='*130)
    print(f"{'Job ID':<38} {'Status':<12} {'Created':<20} {'Duration':<10} {'Step':<40}")
    print('-'*130)

    for job in jobs:
        job_id = job['job_id'][:36]
        status = job['status']
        created = job['created_at'][:19] if job['created_at'] else 'N/A'
        duration = f"{job['actual_duration']}s" if job.get('actual_duration') else 'N/A'
        step = (job.get('current_step') or job.get('error_message') or '')[:38]

        emoji = {
            'pending': '⏳',
            'processing': '⚙️',
            'completed': '✅',
            'failed': '❌'
        }.get(status, '❓')

        print(f"{job_id:<38} {emoji} {status:<10} {created:<20} {duration:<10} {step:<40}")

    print('='*130 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Query job statistics from SQLite database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Show general statistics
  %(prog)s --status pending          # Show pending jobs
  %(prog)s --recent 20               # Show last 20 jobs
  %(prog)s --errors                  # Show error analysis
  %(prog)s --distribution            # Show time distribution
  %(prog)s --job JOB_ID              # Show specific job details
  %(prog)s --export stats.json       # Export stats to JSON
  %(prog)s --cleanup 30              # Cleanup jobs older than 30 days (dry-run)
  %(prog)s --cleanup 30 --confirm    # Actually delete old jobs
        """
    )

    parser.add_argument('--db', default=DEFAULT_DB_PATH,
                        help=f'Database path (default: {DEFAULT_DB_PATH})')
    parser.add_argument('--status', choices=['pending', 'processing', 'completed', 'failed'],
                        help='Filter jobs by status')
    parser.add_argument('--recent', type=int, metavar='N',
                        help='Show N most recent jobs')
    parser.add_argument('--errors', action='store_true',
                        help='Show error analysis')
    parser.add_argument('--distribution', action='store_true',
                        help='Show time distribution')
    parser.add_argument('--job', metavar='JOB_ID',
                        help='Show details for specific job')
    parser.add_argument('--export', metavar='FILE',
                        help='Export statistics to JSON file')
    parser.add_argument('--cleanup', type=int, metavar='DAYS',
                        help='Cleanup jobs older than N days')
    parser.add_argument('--confirm', action='store_true',
                        help='Confirm cleanup (without this, dry-run only)')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = JobStatsAnalyzer(args.db)

    # Handle specific job details
    if args.job:
        job = analyzer.get_job_details(args.job)
        if job:
            print(f"\n{'='*70}")
            print(f" 🔍 JOB DETAILS: {args.job}")
            print('='*70)
            print(json.dumps(job, indent=2, ensure_ascii=False))
            print('='*70 + '\n')
        else:
            print(f"\n❌ Job not found: {args.job}\n")
        return

    # Handle cleanup
    if args.cleanup:
        result = analyzer.cleanup_old_jobs(args.cleanup, dry_run=not args.confirm)
        print(f"\n🗑️  CLEANUP RESULTS")
        print(f"   Jobs to delete: {result['jobs_to_delete']}")
        print(f"   Cutoff date:    {result['cutoff_date']}")
        print(f"   Mode:           {'DRY-RUN (use --confirm to actually delete)' if result['dry_run'] else 'DELETED'}\n")
        return

    # Get general stats
    stats = analyzer.get_total_stats()

    # Export if requested
    if args.export:
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'database': args.db,
            'stats': stats
        }

        if args.errors:
            export_data['errors'] = analyzer.get_error_analysis()

        if args.distribution:
            export_data['distribution'] = analyzer.get_time_distribution()

        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Statistics exported to: {args.export}\n")
        return

    # Show general stats (default)
    print_stats_summary(stats)

    # Show filtered results
    if args.status:
        jobs = analyzer.get_jobs_by_status(args.status)
        print_jobs_table(jobs, f"{args.status.upper()} Jobs")

    elif args.recent:
        jobs = analyzer.get_recent_jobs(args.recent)
        print_jobs_table(jobs, f"Last {args.recent} Jobs")

    if args.errors:
        errors = analyzer.get_error_analysis()
        print(f"\n{'='*70}")
        print(f" ❌ ERROR ANALYSIS")
        print('='*70)
        print(f"\n   Total Failures: {errors['total_failures']}\n")

        if errors['common_errors']:
            print("   Most Common Errors:")
            for i, err in enumerate(errors['common_errors'], 1):
                print(f"   {i}. [{err['count']}x] {err['error'][:60]}")
        print('='*70 + '\n')

    if args.distribution:
        dist = analyzer.get_time_distribution()
        print(f"\n{'='*70}")
        print(f" 📅 TIME DISTRIBUTION")
        print('='*70)

        print("\n   By Hour:")
        for hour, count in sorted(dist['by_hour'].items()):
            bar = '█' * (count // 5 + 1)
            print(f"   {hour}:00  {bar} {count}")

        print("\n   By Day of Week:")
        for day, count in dist['by_day_of_week'].items():
            bar = '█' * (count // 10 + 1)
            print(f"   {day:10}  {bar} {count}")

        print('='*70 + '\n')


if __name__ == '__main__':
    main()
