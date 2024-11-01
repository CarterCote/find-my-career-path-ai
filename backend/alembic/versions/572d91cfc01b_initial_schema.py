"""initial schema

Revision ID: 572d91cfc01b
Revises: 
Create Date: 2024-03-19 12:34:56.789012

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '572d91cfc01b'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('job_postings',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('job_id', sa.BigInteger(), nullable=True),
    sa.Column('company_name', sa.Text(), nullable=True),
    sa.Column('title', sa.Text(), nullable=True),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('max_salary', sa.Float(), nullable=True),
    sa.Column('pay_period', sa.String(length=50), nullable=True),
    sa.Column('location', sa.Text(), nullable=True),
    sa.Column('company_id', sa.Float(), nullable=True),
    sa.Column('views', sa.Float(), nullable=True),
    sa.Column('med_salary', sa.Float(), nullable=True),
    sa.Column('min_salary', sa.Float(), nullable=True),
    sa.Column('formatted_work_type', sa.String(length=50), nullable=True),
    sa.Column('description_embedding', Vector(dim=768), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('job_id')
    )
    op.create_table('user_profiles',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.String(), nullable=False),
    sa.Column('core_values', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('work_culture', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('skills', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('top_six', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('preference_rankings', sa.JSON(), nullable=True),
    sa.Column('additional_interests', sa.Text(), nullable=True),
    sa.Column('background', sa.Text(), nullable=True),
    sa.Column('goals', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_id')
    )
    op.create_table('career_recommendations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.String(), nullable=False),
    sa.Column('career_title', sa.String(), nullable=False),
    sa.Column('reasoning', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['user_profiles.session_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('chat_histories',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.String(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('is_user', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['user_profiles.session_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('chat_histories')
    op.drop_table('career_recommendations')
    op.drop_table('user_profiles')
    op.drop_table('job_postings')
    # ### end Alembic commands ###
