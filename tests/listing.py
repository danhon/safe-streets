"""What the pipeline and batcher tests read back: the newest passes and how
many there are (the table's old `passes_page(1, 10)`)."""


async def first_page(db, size: int = 10):
    return await db.passes_list(limit=size), await db.count_passes()
