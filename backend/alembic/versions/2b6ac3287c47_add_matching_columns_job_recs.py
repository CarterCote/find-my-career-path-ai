"""add_matching_columns_job_recs

Revision ID: 2b6ac3287c47
Revises: 7c442e62af4a
Create Date: 2024-11-28 12:15:21.679948

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '2b6ac3287c47'
down_revision: Union[str, None] = '7c442e62af4a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to job_recommendations
    with op.batch_alter_table('job_recommendations') as batch_op:
        batch_op.add_column(sa.Column('matching_skills', sa.ARRAY(sa.String()), nullable=True))
        batch_op.add_column(sa.Column('matching_culture', sa.ARRAY(sa.String()), nullable=True))
        batch_op.add_column(sa.Column('location', sa.String(255), nullable=True))

def downgrade() -> None:
    # Remove the columns if needed to rollback
    with op.batch_alter_table('job_recommendations') as batch_op:
        batch_op.drop_column('matching_skills')
        batch_op.drop_column('matching_culture')
        batch_op.drop_column('location')