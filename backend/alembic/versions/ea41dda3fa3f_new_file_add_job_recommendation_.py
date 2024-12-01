"""new_file_add_job_recommendation_preference_version

Revision ID: ea41dda3fa3f
Revises: 1b8448987a8b
Create Date: 2024-11-30 18:22:13.712043

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ea41dda3fa3f'
down_revision: Union[str, None] = '1b8448987a8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add preference_version to job_recommendations table
    op.add_column('job_recommendations', sa.Column('preference_version', sa.Integer(), server_default='1', nullable=False))

def downgrade() -> None:
    op.drop_column('job_recommendations', 'preference_version')