from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from ..config import get_settings

# Configure logging
import logging
import sys
from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

settings = get_settings()

app = Celery(
    'backend',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=[
        'backend.src.celery_worker.tasks',
        'backend.src.utils.job_search',
    ]
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_routes={
        'backend.src.celery_worker.tasks.*': {'queue': 'job_search'},
        'backend.src.utils.job_search.*': {'queue': 'job_search'},
    },
    task_default_queue='job_search',
    worker_prefetch_multiplier=1,
    task_time_limit=600,
    task_soft_time_limit=300,
)