"""fix_career_recommendations_fk

Revision ID: 91fba55415cb
Revises: 50210e984fd2
Create Date: 2024-11-20 17:28:46.902005

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '91fba55415cb'
down_revision: Union[str, None] = '50210e984fd2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the career_recommendations table and recreate it
    op.drop_table('career_recommendations')
    
    op.create_table('career_recommendations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_session_id', sa.String(), nullable=False),
        sa.Column('career_title', sa.String(255), nullable=False),
        sa.Column('career_field', sa.String(255)),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('skills_required', sa.ARRAY(sa.String())),
        sa.Column('growth_potential', sa.Text()),
        sa.Column('match_score', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_session_id'], ['user_profiles.user_session_id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('career_recommendations')
    
    op.create_table('career_recommendations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('career_title', sa.String(255), nullable=False),
        sa.Column('career_field', sa.String(255)),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('skills_required', sa.ARRAY(sa.String())),
        sa.Column('growth_potential', sa.Text()),
        sa.Column('match_score', sa.Float()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['user_profiles.session_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
