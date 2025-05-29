from uuid import UUID, uuid4
from pydantic import BaseModel
from sqlmodel import SQLModel, Field
from datetime import datetime, timezone
import time
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship


class ChatSession(SQLModel, table=True):
    sid: UUID = Field(primary_key=True, default_factory=uuid4)
    uid: str = Field(index=True)
    agent: str
    createdAt: int = Field(
        default_factory=lambda: int(time.time() * 1000))
    access: str = Field(default="private")
    messages: List["Message"] = Relationship(back_populates="session")


class Message(SQLModel, table=True):
    id: UUID = Field(primary_key=True, default_factory=uuid4)
    sid: UUID = Field(foreign_key="chatsession.sid")
    type: str
    authorId: str
    createdAt: int = Field(default_factory=lambda: int(time.time() * 1000))
    text: str

    session: Optional[ChatSession] = Relationship(back_populates="messages")
