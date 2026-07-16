import asyncio
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from scientific_agent.config import Settings
from scientific_agent.knowledge import KnowledgeLibrary
from scientific_agent.knowledge_indexing import (
    KnowledgeSemanticIndexer,
    KnowledgeIndexService,
    TextSemanticBatch,
    VisualSemanticBatch,
)
from scientific_agent.knowledge_visuals import (
    extract_visual_candidates,
    register_document_visuals,
)


class FakeLibrary:
    def __init__(self, document: dict, chunks: list[dict], visuals: list[dict]):
        self.document = {
            "published": False,
            "semantic_status": "pending",
            **document,
        }
        self._chunks = chunks
        self._visuals = visuals
        self.applied = None
        self.failure = None

    def get_document(self, document_id, *, include_retired=False):
        assert document_id == self.document["id"]
        return self.document

    def semantic_source_chunks(self, document_id):
        assert document_id == self.document["id"]
        return self._chunks

    def visual_assets(self, document_id):
        assert document_id == self.document["id"]
        return self._visuals

    def apply_semantic_index(self, document_id, **values):
        assert document_id == self.document["id"]
        self.applied = values

    def mark_semantic_index_failed(self, document_id, error_type, **_):
        assert document_id == self.document["id"]
        self.failure = error_type


def test_text_knowledge_is_indexed_by_qwen_without_gemma(tmp_path):
    calls = []

    async def requester(endpoint, *, payload, output_type, image_paths=(), **kwargs):
        calls.append((endpoint.model, payload, image_paths, kwargs))
        assert output_type is TextSemanticBatch
        assert not image_paths
        return TextSemanticBatch(
            entries=[
                {
                    "chunk_ordinal": item["chunk_ordinal"],
                    "concepts": ["overall survival"],
                    "synonyms": ["OS"],
                }
                for item in payload["chunks"]
            ]
        )

    library = FakeLibrary(
        {
            "id": "doc",
            "title": "Trial",
            "source_type": "primary_study",
            "extractor": "markdown",
        },
        [
            {
                "id": "chunk",
                "ordinal": 0,
                "sha256": "a" * 64,
                "content": "Exact source text.",
            }
        ],
        [],
    )
    settings = Settings()
    result = asyncio.run(
        KnowledgeSemanticIndexer(settings, requester=requester).index_document(
            library, "doc"
        )
    )

    assert result == library.document
    assert [item[0] for item in calls] == [settings.qwen.model]
    assert library.applied["text_entries"][0]["chunk_id"] == "chunk"
    assert library.applied["visual_entries"] == []
    assert library.applied["metadata"]["routing"] == {
        "text": "qwen",
        "visual": "gemma-only-if-images",
    }
    assert calls[0][3]["timeout"] == settings.qwen.request_timeout_seconds
    assert callable(calls[0][3]["on_visible_text"])


def test_only_actual_visuals_are_sent_to_gemma(tmp_path):
    image_path = tmp_path / "visual.jpg"
    Image.new("RGB", (80, 60), "white").save(image_path)
    visual_id = "kv-" + "1" * 24
    calls = []

    async def requester(endpoint, *, payload, output_type, image_paths=(), **kwargs):
        calls.append((endpoint.model, payload, image_paths))
        assert output_type is VisualSemanticBatch
        assert image_paths == (image_path,)
        assert payload["visual_input_order"] == [0]
        assert kwargs["timeout"] == Settings().gemma.request_timeout_seconds
        assert callable(kwargs["on_visible_text"])
        return VisualSemanticBatch(
            entries=[{"visual_index": 0, "visible_terms": ["Kaplan-Meier curve"]}]
        )

    library = FakeLibrary(
        {
            "id": "image",
            "title": "Figure",
            "source_type": "other",
            "extractor": "image-metadata-only",
        },
        [
            {
                "id": "metadata",
                "ordinal": 0,
                "sha256": "b" * 64,
                "content": "File metadata.",
            }
        ],
        [
            {
                "id": visual_id,
                "sha256": "c" * 64,
                "source_label": "uploaded image",
                "path": str(image_path),
            }
        ],
    )
    settings = Settings()
    asyncio.run(
        KnowledgeSemanticIndexer(settings, requester=requester).index_document(
            library, "image"
        )
    )

    assert [item[0] for item in calls] == [settings.gemma.model]
    assert library.applied["text_entries"] == []
    assert library.applied["visual_entries"][0]["visual_id"] == visual_id


def test_visual_extraction_normalizes_uploaded_and_office_images(tmp_path):
    raster = tmp_path / "figure.png"
    Image.new("RGB", (320, 180), "navy").save(raster)
    direct = extract_visual_candidates(raster, tmp_path / "direct")

    presentation = tmp_path / "slides.pptx"
    with zipfile.ZipFile(presentation, "w") as archive:
        archive.write(raster, "ppt/media/image1.png")
    embedded = extract_visual_candidates(presentation, tmp_path / "office")

    assert len(direct) == 1
    assert direct[0]["extractor"] == "direct-raster-normalizer"
    assert Path(direct[0]["path"]).is_file()
    assert len(embedded) == 1
    assert embedded[0]["source_label"] == "ppt/media/image1.png"
    assert embedded[0]["extractor"] == "ooxml-media-normalizer"


def test_real_library_routes_text_to_qwen_and_registered_image_to_gemma(tmp_path):
    library = KnowledgeLibrary(tmp_path / "knowledge", "test", "https://bench.test")
    image = tmp_path / "figure.png"
    Image.new("RGB", (160, 100), "navy").save(image)
    with image.open("rb") as handle:
        document = library.ingest(
            image.name,
            handle,
            image.stat().st_size,
            title="Uploaded scientific figure",
            semantic_pending=True,
        )
    registered = register_document_visuals(library, document["id"])
    calls = []

    async def requester(endpoint, *, output_type, image_paths=(), **kwargs):
        calls.append((endpoint.model, image_paths, kwargs["timeout"]))
        assert output_type is VisualSemanticBatch
        return VisualSemanticBatch(
            entries=[{"visual_index": 0, "visible_terms": ["survival curve"]}]
        )

    asyncio.run(
        KnowledgeSemanticIndexer(Settings(), requester=requester).index_document(
            library, document["id"]
        )
    )

    published = library.get_document(document["id"])
    assert registered and published["published"]
    assert published["semantic_status"] == "ready"
    assert published["chunk_count"] == 0
    assert [call[0] for call in calls] == [Settings().gemma.model]


def test_office_visual_extraction_rejects_extreme_compression_ratio(tmp_path):
    presentation = tmp_path / "compressed.pptx"
    with zipfile.ZipFile(
        presentation, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("ppt/media/image1.png", b"0" * (2 * 1024 * 1024))

    assert extract_visual_candidates(presentation, tmp_path / "output") == []


def test_persistent_worker_publishes_candidate_and_job_atomically(tmp_path):
    library = KnowledgeLibrary(tmp_path / "knowledge", "test", "https://bench.test")
    source = b"A randomized trial measured exact disease-free survival."
    document = library.ingest(
        "trial.md",
        BytesIO(source),
        len(source),
        title="Trial",
        source_type="primary_study",
        semantic_pending=True,
    )

    async def requester(endpoint, *, payload, output_type, **kwargs):
        assert endpoint.model == Settings().qwen.model
        assert output_type is TextSemanticBatch
        assert kwargs["timeout"] == Settings().qwen.request_timeout_seconds
        return TextSemanticBatch(
            entries=[
                {
                    "chunk_ordinal": item["chunk_ordinal"],
                    "concepts": ["disease-free survival"],
                    "synonyms": ["DFS"],
                }
                for item in payload["chunks"]
            ]
        )

    async def run_worker():
        service = KnowledgeIndexService(
            library,
            KnowledgeSemanticIndexer(Settings(), requester=requester),
            scientific_work_active=lambda: False,
            poll_seconds=0.01,
        )
        job = service.enqueue(document["id"], "upload")
        await service.start()
        try:
            for _ in range(300):
                current = library.get_index_job(job["id"])
                if current["status"] not in {
                    "queued",
                    "running",
                    "cancel_requested",
                }:
                    return current
                await asyncio.sleep(0.01)
            raise AssertionError("knowledge index worker did not terminate")
        finally:
            await service.close()

    final_job = asyncio.run(run_worker())

    published = library.get_document(document["id"])
    assert final_job["status"] == "succeeded"
    assert published["published"]
    assert published["semantic_status"] == "ready"
    assert any(
        event["status"] == "succeeded"
        for event in library.list_index_events(final_job["id"])
    )


def test_shutdown_preserves_pending_candidate_and_restart_recovers_job(tmp_path):
    library = KnowledgeLibrary(tmp_path / "knowledge", "test", "https://bench.test")
    source = b"A durable source must survive controller shutdown during inference."
    document = library.ingest(
        "durable.md",
        BytesIO(source),
        len(source),
        title="Durable candidate",
        semantic_pending=True,
    )

    async def scenario():
        request_started = asyncio.Event()
        never_finishes = asyncio.Event()

        async def blocking_requester(endpoint, *, payload, output_type, **kwargs):
            del endpoint, payload, output_type, kwargs
            request_started.set()
            await never_finishes.wait()
            raise AssertionError("unreachable")

        first = KnowledgeIndexService(
            library,
            KnowledgeSemanticIndexer(Settings(), requester=blocking_requester),
            scientific_work_active=lambda: False,
            poll_seconds=0.01,
        )
        job = first.enqueue(document["id"], "upload")
        await first.start()
        await asyncio.wait_for(request_started.wait(), timeout=3)
        await first.close()

        interrupted_job = library.get_index_job(job["id"])
        interrupted_document = library.get_document(document["id"])
        assert interrupted_job["status"] == "running"
        assert interrupted_document["semantic_status"] == "pending"
        assert interrupted_document["published"] is False

        async def successful_requester(endpoint, *, payload, output_type, **kwargs):
            del endpoint, kwargs
            assert output_type is TextSemanticBatch
            return TextSemanticBatch(
                entries=[
                    {
                        "chunk_ordinal": item["chunk_ordinal"],
                        "concepts": ["durable restart recovery"],
                    }
                    for item in payload["chunks"]
                ]
            )

        restarted = KnowledgeIndexService(
            library,
            KnowledgeSemanticIndexer(Settings(), requester=successful_requester),
            scientific_work_active=lambda: False,
            poll_seconds=0.01,
        )
        await restarted.start()
        try:
            for _ in range(300):
                current = library.get_index_job(job["id"])
                if current["status"] == "succeeded":
                    return current
                await asyncio.sleep(0.01)
            raise AssertionError("recovered knowledge job did not finish")
        finally:
            await restarted.close()

    final_job = asyncio.run(scenario())

    assert final_job["attempt"] == 2
    assert library.get_document(document["id"])["published"] is True
    assert [
        event["status"] for event in library.list_index_events(final_job["id"])
    ].count("queued") == 2


def test_user_cancellation_fails_candidate_instead_of_shutdown_recovery(tmp_path):
    library = KnowledgeLibrary(tmp_path / "knowledge", "test", "https://bench.test")
    source = b"A user-cancelled semantic indexing source."
    document = library.ingest(
        "cancel.md",
        BytesIO(source),
        len(source),
        title="Cancelled candidate",
        semantic_pending=True,
    )

    async def scenario():
        request_started = asyncio.Event()

        async def cancellable_requester(endpoint, *, cancel_event, **kwargs):
            del endpoint, kwargs
            request_started.set()
            while not cancel_event.is_set():
                await asyncio.sleep(0.005)
            raise asyncio.CancelledError

        service = KnowledgeIndexService(
            library,
            KnowledgeSemanticIndexer(Settings(), requester=cancellable_requester),
            scientific_work_active=lambda: False,
            poll_seconds=0.01,
        )
        job = service.enqueue(document["id"], "upload")
        await service.start()
        try:
            await asyncio.wait_for(request_started.wait(), timeout=3)
            service.request_cancel(job["id"])
            for _ in range(300):
                current = library.get_index_job(job["id"])
                if current["status"] == "cancelled":
                    return current
                await asyncio.sleep(0.01)
            raise AssertionError("user cancellation did not finish")
        finally:
            await service.close()

    final_job = asyncio.run(scenario())

    candidate = library.get_document(document["id"])
    assert final_job["status"] == "cancelled"
    assert candidate["semantic_status"] == "failed"
    assert candidate["published"] is False
