"""update_chat_and_job_tables

Revision ID: f33bcd0b5c02
Revises: d8aed70c1fb1
Create Date: 2024-11-20 16:15:26.552517

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'f33bcd0b5c02'
down_revision: Union[str, None] = 'd8aed70c1fb1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Drop job_recommendations if it exists
    op.execute('DROP TABLE IF EXISTS job_recommendations CASCADE')

    # 2. Create job_recommendations table
    op.create_table('job_recommendations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('chat_session_id', sa.String(), nullable=False),
        sa.Column('job_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(255)),
        sa.Column('company_name', sa.String(255)),
        sa.Column('match_score', sa.Float()),
        sa.Column('recommendation_type', sa.String(50)),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['postings.job_id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['user_profiles.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    # Drop job_recommendations table
    op.drop_table('job_recommendations')