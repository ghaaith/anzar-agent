"""add checkpoints and tasks tables

Revision ID: c9d4e2f7a1b3
Revises: b3f7c1e2a9d4
Create Date: 2026-09-03 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c9d4e2f7a1b3'
down_revision: Union[str, None] = 'b3f7c1e2a9d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'checkpoints',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('ordinal', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('workspace_path', sa.String(length=500), nullable=False),
        sa.Column('git_commit', sa.String(length=64), nullable=True),
        sa.Column('snapshot_path', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('files_manifest', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_checkpoints_workspace_path', 'checkpoints', ['workspace_path'])
    op.create_table(
        'tasks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('checkpoint_id', sa.UUID(), nullable=True),
        sa.Column('workspace_path', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('files_created', sa.Text(), nullable=True),
        sa.Column('files_modified', sa.Text(), nullable=True),
        sa.Column('files_deleted', sa.Text(), nullable=True),
        sa.Column('verification_status', sa.String(length=20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['checkpoint_id'], ['checkpoints.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('tasks')
    op.drop_index('ix_checkpoints_workspace_path', table_name='checkpoints')
    op.drop_table('checkpoints')
