"""add evaluation data to job recommendations

Revision ID: e0f98b3e36bf
Revises: ea41dda3fa3f
Create Date: 2024-11-30 23:48:16.882587

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = 'e0f98b3e36bf'
down_revision: Union[str, None] = 'ea41dda3fa3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('job_recommendations', 
                  sa.Column('evaluation_data', JSONB, nullable=True))

def downgrade() -> None:
    op.drop_column('job_recommendations', 'evaluation_data')