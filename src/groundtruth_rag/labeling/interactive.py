"""Terminal passage review; final labels require human confirmation."""

import typer

from groundtruth_rag.labeling.candidates import Candidate, discover_candidates, search_corpus
from groundtruth_rag.labeling.session import LabelSession
from groundtruth_rag.retrieval.base import Retriever
from groundtruth_rag.schemas import DocumentChunk


def _show_candidates(candidates: list[Candidate]) -> None:
    if not candidates:
        typer.echo("No candidates. Use 'search <phrase>' to inspect the corpus.")
    for number, candidate in enumerate(candidates, 1):
        chunk = candidate.chunk
        ranks = ", ".join(f"{hit.retriever} rank {hit.rank} score {hit.score:.4f}"
                          for hit in candidate.matches) or "corpus search"
        typer.echo(f"\n[{number}] {ranks}\nSource: {chunk.source} | Page: {chunk.page or 'n/a'} | "
                   f"Chunk index: {chunk.metadata.get('chunk_index', 'n/a')}\n"
                   f"Chunk ID: {chunk.chunk_id}\nText: {chunk.text[:400]}")


def _ids_from_selection(value: str, candidates: list[Candidate],
                        by_id: dict[str, DocumentChunk]) -> list[str]:
    ids: list[str] = []
    for token in value.replace(",", " ").split():
        if token.isdecimal():
            number = int(token)
            if number < 1 or number > len(candidates):
                raise ValueError(f"candidate number out of range: {token}")
            ids.append(candidates[number - 1].chunk.chunk_id)
        else:
            if token not in by_id:
                raise ValueError(f"chunk ID absent from corpus: {token}")
            ids.append(token)
    if not ids:
        raise ValueError("select at least one chunk")
    return ids


def run_label_loop(session: LabelSession, chunks: list[DocumentChunk],
                   dense: Retriever, bm25: Retriever, depth: int) -> None:
    if depth <= 0:
        raise ValueError("candidate depth must be positive")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    index = next((i for i, item in enumerate(session.state.questions)
                  if item.status != "verified"), len(session.state.questions))
    if index == len(session.state.questions):
        typer.echo("All draft questions are verified.")
        return
    while 0 <= index < len(session.state.questions):
        record = session.state.questions[index]
        current_depth = depth
        candidates = discover_candidates(record.query, chunks, dense, bm25, current_depth)
        typer.echo(f"\n{index + 1}/{len(session.state.questions)} {record.query_id}: {record.query}")
        typer.echo(f"Status: {record.status}; selected: {', '.join(record.selected_chunk_ids) or 'none'}")
        _show_candidates(candidates)
        typer.echo("Actions: select/add <numbers or IDs>, inspect <number or ID>, more, search <phrase>, "
                   "answer <text>, notes <text>, tags <comma list>, confirm, skip, prev, next, quit")
        while True:
            action = typer.prompt("Action").strip()
            command, _, value = action.partition(" ")
            try:
                if command == "quit":
                    return
                if command in {"next", "prev"}:
                    index += 1 if command == "next" else -1
                    index = max(0, min(index, len(session.state.questions) - 1))
                    break
                if command == "more":
                    current_depth = min(len(chunks), current_depth * 2)
                    candidates = discover_candidates(record.query, chunks, dense, bm25, current_depth)
                    _show_candidates(candidates)
                    continue
                if command == "search":
                    candidates = search_corpus(chunks, value)
                    _show_candidates(candidates)
                    continue
                if command == "inspect":
                    selected = _ids_from_selection(value, candidates, by_id)
                    for chunk_id in selected:
                        chunk = by_id[chunk_id]
                        typer.echo(f"\n{chunk.source} page {chunk.page or 'n/a'} | {chunk_id}\n{chunk.text}")
                    continue
                if command == "confirm":
                    if typer.confirm(f"Verify {record.query_id} with {len(record.selected_chunk_ids)} selected chunk(s)?",
                                     default=False):
                        session.verify(record.query_id, confirmed=True)
                        typer.echo(f"Verified {record.query_id}")
                        index += 1
                        break
                    continue
                if command in {"select", "add", "answer", "notes", "tags", "skip"}:
                    editing_verified = record.status == "verified"
                    if editing_verified and not typer.confirm(
                        f"Edit verified answer {record.query_id}? It will require reconfirmation.", default=False
                    ):
                        continue
                    if command in {"select", "add"}:
                        chosen = _ids_from_selection(value, candidates, by_id)
                        if command == "add":
                            chosen = list(dict.fromkeys([*record.selected_chunk_ids, *chosen]))
                        session.select(record.query_id, chosen, allow_verified=editing_verified)
                    elif command == "answer":
                        session.edit(record.query_id, answer=value, allow_verified=editing_verified)
                    elif command == "notes":
                        session.edit(record.query_id, notes=value, allow_verified=editing_verified)
                    elif command == "tags":
                        session.edit(record.query_id, tags=[tag.strip() for tag in value.split(",")]
                                     if value.strip() else [],
                                     allow_verified=editing_verified)
                    else:
                        session.skip(record.query_id, allow_verified=editing_verified)
                        index += 1
                        break
                    typer.echo(f"Saved {record.query_id}: {record.status}; "
                               f"selected {', '.join(record.selected_chunk_ids) or 'none'}")
                    continue
                typer.echo("Unknown action.")
            except ValueError as exc:
                typer.echo(f"Action rejected: {exc}")
