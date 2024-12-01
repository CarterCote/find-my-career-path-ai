"""add_preference_tracking_columns

Revision ID: 1b8448987a8b
Revises: dd24ac4644e4
Create Date: 2024-11-30 18:03:30.729461

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '1b8448987a8b'
down_revision: Union[str, None] = 'dd24ac4644e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('user_profiles', sa.Column('preference_version', sa.Integer(), server_default='1', nullable=False))
    op.add_column('user_profiles', sa.Column('last_preference_update', sa.DateTime(), server_default=sa.text('now()'), nullable=False))

def downgrade() -> None:
    op.drop_column('user_profiles', 'last_preference_update')
    op.drop_column('user_profiles', 'preference_version')
    # ### end Alembic commands ###
