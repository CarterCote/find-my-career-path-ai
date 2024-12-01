"""remove_unused_json_fields

Revision ID: dd24ac4644e4
Revises: 2b6ac3287c47
Create Date: 2024-11-30 17:39:40.607631

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'dd24ac4644e4'
down_revision: Union[str, None] = '2b6ac3287c47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove unused JSON columns from user_profiles
    with op.batch_alter_table('user_profiles') as batch_op:
        batch_op.drop_column('career_recommendations')
        batch_op.drop_column('job_recommendations')

def downgrade() -> None:
    # Add back JSON columns if needed
    with op.batch_alter_table('user_profiles') as batch_op:
        batch_op.add_column(sa.Column('career_recommendations', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('job_recommendations', sa.JSON(), nullable=True))