"""add max_steps to settings

Revision ID: b3f7c1e2a9d4
Revises: 38962df355c7
Create Date: 2026-08-14 15:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b3f7c1e2a9d4'
down_revision: Union[str, None] = '38962df355c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('settings', sa.Column('max_steps', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('settings', 'max_steps')
