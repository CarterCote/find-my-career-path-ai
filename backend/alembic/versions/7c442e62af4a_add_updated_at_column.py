"""add_updated_at_column

Revision ID: 7c442e62af4a
Revises: 91fba55415cb
Create Date: 2024-11-20 17:34:54.445931

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '7c442e62af4a'
down_revision: Union[str, None] = '91fba55415cb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add updated_at column to user_profiles
    op.add_column('user_profiles',
        sa.Column('updated_at', 
                 sa.DateTime(), 
                 server_default=sa.text('now()'),
                 nullable=False)
    )

def downgrade() -> None:
    op.drop_column('user_profiles', 'updated_at')