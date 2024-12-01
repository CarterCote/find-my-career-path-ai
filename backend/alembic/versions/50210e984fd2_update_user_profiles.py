"""update_user_profiles

Revision ID: 50210e984fd2
Revises: f33bcd0b5c02
Create Date: 2024-11-20 17:22:34.095255

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '50210e984fd2'
down_revision: Union[str, None] = 'f33bcd0b5c02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add new columns to user_profiles
    with op.batch_alter_table('user_profiles') as batch_op:
        # First rename session_id to user_session_id
        batch_op.alter_column('session_id', 
                            new_column_name='user_session_id',
                            existing_type=sa.String())
        
        # Add new columns
        batch_op.add_column(sa.Column('career_recommendations', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('job_recommendations', sa.JSON(), nullable=True))

    # 2. Update CareerRecommendation foreign key
    with op.batch_alter_table('career_recommendations') as batch_op:
        batch_op.drop_constraint('career_recommendations_session_id_fkey', type_='foreignkey')
        batch_op.alter_column('session_id',
                            new_column_name='user_session_id',
                            existing_type=sa.String())
        batch_op.create_foreign_key(
            'career_recommendations_user_session_id_fkey',
            'user_profiles',
            ['user_session_id'],
            ['user_session_id']
        )


def downgrade() -> None:
    # 1. Revert CareerRecommendation changes
    with op.batch_alter_table('career_recommendations') as batch_op:
        batch_op.drop_constraint('career_recommendations_user_session_id_fkey', type_='foreignkey')
        batch_op.alter_column('user_session_id',
                            new_column_name='session_id',
                            existing_type=sa.String())
        batch_op.create_foreign_key(
            'career_recommendations_session_id_fkey',
            'user_profiles',
            ['session_id'],
            ['session_id']
        )

    # 2. Revert user_profiles changes
    with op.batch_alter_table('user_profiles') as batch_op:
        batch_op.drop_column('job_recommendations')
        batch_op.drop_column('career_recommendations')
        batch_op.alter_column('user_session_id',
                            new_column_name='session_id',
                            existing_type=sa.String())
