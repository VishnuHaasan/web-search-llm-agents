import re

class TextChunker:

    def __init__(self, max_chunk_size: int, chunk_overlap: int):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_by_spaces(self, text: str) -> list[str]:
        out = []
        curr = ""
        for s in text.split():
            if len(s) > self.max_chunk_size:
                if curr:
                    out.append(curr)
                    curr = ""
                out.extend(s[i:i + self.max_chunk_size] for i in range(0, len(s), self.max_chunk_size))
                continue
            new_curr = f"{curr} {s}" if curr else s
            if len(new_curr) > self.max_chunk_size:
                out.append(curr)
                curr = s
            else:
                curr = new_curr
        if curr:
            out.append(curr)

        return out


    def _split_long_chunks(self, chunk: str) -> list[str]:
        if len(chunk) <= self.max_chunk_size:
            return [chunk]
        out = []
        for sentence in re.split(r"(?<=[.!?])\s+", chunk):
            if not sentence:
                continue

            if len(sentence) > self.max_chunk_size:
                out.extend(self._split_by_spaces(sentence))
            else:
                out.append(sentence)
        return out

    @staticmethod
    def _joined_len(lines: list[str]) -> int:
        return sum(len(line) for line in lines) + len(lines) - 1

    def _tail(self, box: list[str]) -> list[str]:
        tail = []
        total = 0
        for line in reversed(box):
            added = len(line) + (1 if tail else 0)
            if total + added > self.chunk_overlap:
                break
            tail.insert(0, line)
            total += added

        return tail

    def _pack_chunks(self, pieces: list[str]) -> list[str]:
        chunks, box = [], []
        for piece in pieces:
            if box and self._joined_len(box + [piece]) > self.max_chunk_size:
                chunks.append("\n".join(box))
                box = self._tail(box)
                while box and self._joined_len(box + [piece]) > self.max_chunk_size:
                    box.pop(0)
            box.append(piece)
        if box:
            chunks.append("\n".join(box))
        return chunks

    def chunk_text(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        chunked = []
        for line in lines:
            chunked.extend(self._split_long_chunks(line))
        return self._pack_chunks(chunked)
