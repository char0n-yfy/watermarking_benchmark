from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.resources import IMAGE_EXTS, iter_image_paths


COMPACT_SAMPLE_COUNT = 1000


@dataclass(frozen=True)
class DatasetCatalogEntry:
    id: str
    name: str
    name_zh: str
    category: str
    category_zh: str
    description: str
    description_zh: str
    source_url: str
    official_total_images: int | None = None
    manifest_url: str | None = None
    compact_uses_root: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "nameZh": self.name_zh,
            "category": self.category,
            "categoryZh": self.category_zh,
            "description": self.description,
            "descriptionZh": self.description_zh,
            "sourceUrl": self.source_url,
            "officialTotalImages": self.official_total_images,
            "manifestUrl": self.manifest_url,
            "compactSampleCount": COMPACT_SAMPLE_COUNT,
            "compactUsesRoot": self.compact_uses_root,
        }


DATASET_CATALOG: tuple[DatasetCatalogEntry, ...] = (
    DatasetCatalogEntry(
        id="ms-coco",
        name="MS COCO",
        name_zh="MS COCO",
        category="natural-benchmark",
        category_zh="基础自然图像基准",
        description="Microsoft COCO provides everyday-scene photos with bounding boxes, segmentation masks, and captions.",
        description_zh="微软 COCO 日常场景图像数据集，广泛用于目标检测、实例分割与图像描述。",
        source_url="https://cocodataset.org/#download",
        official_total_images=123_287,
    ),
    DatasetCatalogEntry(
        id="imagenet",
        name="ImageNet",
        name_zh="ImageNet",
        category="natural-benchmark",
        category_zh="基础自然图像基准",
        description="ImageNet ILSVRC is a large-scale hierarchical image database for object recognition research.",
        description_zh="ImageNet ILSVRC 大规模层次化图像数据库，是物体识别与分类研究的基础基准。",
        source_url="https://image-net.org/download.php",
        official_total_images=1_431_167,
    ),
    DatasetCatalogEntry(
        id="diffusiondb",
        name="DiffusionDB",
        name_zh="DiffusionDB",
        category="aigc",
        category_zh="AIGC 图像",
        description="DiffusionDB archives Stable Diffusion outputs paired with user prompts for studying text-to-image generation.",
        description_zh="DiffusionDB 收录 Stable Diffusion 生成图像及对应提示词，用于研究生成式 AIGC 内容。",
        source_url="https://poloclub.github.io/diffusiondb/",
        official_total_images=14_000_000,
    ),
    DatasetCatalogEntry(
        id="w-bench",
        name="W-Bench",
        name_zh="W-Bench",
        category="aigc",
        category_zh="AIGC 图像",
        description="W-Bench (VINE, ICLR 2025) evaluates watermark robustness under regeneration, editing, and traditional distortions.",
        description_zh="W-Bench（VINE, ICLR 2025）系统评测水印在再生成、图像编辑与传统失真下的鲁棒性。",
        source_url="https://huggingface.co/datasets/Shilin-LU/W-Bench",
        official_total_images=100_000,
    ),
    DatasetCatalogEntry(
        id="4k-benchmark",
        name="4K Benchmark Images",
        name_zh="4K Benchmark Images",
        category="hd-copyright",
        category_zh="高清版权图",
        description="4KLSDB is a native-4K image dataset for high-resolution restoration and text-to-image generation research.",
        description_zh="4KLSDB 原生 4K 图像数据集，面向高分辨率修复与文生图生成研究。",
        source_url="https://4klsdb.github.io/",
        official_total_images=133_468,
    ),
    DatasetCatalogEntry(
        id="flickr2k",
        name="Flickr2K",
        name_zh="Flickr2K",
        category="hd-copyright",
        category_zh="高清版权图",
        description="Flickr2K is a Flickr-sourced 2K-resolution image set released for NTIRE super-resolution research.",
        description_zh="Flickr2K 是 NTIRE 超分挑战赛发布的 Flickr 2K 分辨率图像集合。",
        source_url="https://opendatalab.com/OpenDataLab/Flickr2K",
        official_total_images=2650,
    ),
    DatasetCatalogEntry(
        id="openimages-v7",
        name="OpenImages V7",
        name_zh="OpenImages V7",
        category="open-world",
        category_zh="真实复杂开放世界图片",
        description="Google Open Images V7 is a web-scale dataset with object detection, segmentation, and visual-relationship labels.",
        description_zh="Google Open Images V7 是带检测、分割与视觉关系标注的大规模开放世界图像集。",
        source_url="https://storage.googleapis.com/openimages/web/index.html",
        official_total_images=9_000_000,
    ),
    DatasetCatalogEntry(
        id="mapillary-vistas",
        name="Mapillary Vistas",
        name_zh="Mapillary Vistas",
        category="open-world",
        category_zh="真实复杂开放世界图片",
        description="Mapillary Vistas provides street-level imagery with dense pixel-wise semantic segmentation annotations.",
        description_zh="Mapillary Vistas 提供街景级图像及像素级语义分割标注。",
        source_url="https://www.mapillary.com/dataset/vistas",
        official_total_images=25_000,
    ),
    DatasetCatalogEntry(
        id="doclaynet",
        name="DocLayNet",
        name_zh="DocLayNet",
        category="document",
        category_zh="文档、截图、海报类场景",
        description="DocLayNet (IBM) is a document-layout dataset with human-annotated regions on diverse PDF scans.",
        description_zh="DocLayNet（IBM）是面向多样 PDF 扫描件的文档版面结构标注数据集。",
        source_url="https://github.com/DS4SD/DocLayNet",
        official_total_images=80_863,
    ),
    DatasetCatalogEntry(
        id="publaynet",
        name="PubLayNet",
        name_zh="PubLayNet",
        category="document",
        category_zh="文档、截图、海报类场景",
        description="PubLayNet derives scientific-publication PDF pages into layout-detection training images.",
        description_zh="PubLayNet 将科学出版物 PDF 页面转换为版面检测训练图像。",
        source_url="https://github.com/ibm-aur-nlp/PubLayNet",
        official_total_images=360_000,
    ),
    DatasetCatalogEntry(
        id="shopee-product-matching",
        name="Shopee Product Matching",
        name_zh="Shopee Product Matching",
        category="ecommerce",
        category_zh="电商版权保护",
        description="Kaggle Shopee Product Matching provides e-commerce product photos for duplicate-listing and copyright detection.",
        description_zh="Kaggle 虾皮商品匹配赛数据集，面向电商同款识别与版权保护场景。",
        source_url="https://www.kaggle.com/c/shopee-product-matching/data",
        official_total_images=51_000,
        compact_uses_root=True,
    ),
    DatasetCatalogEntry(
        id="products-10k",
        name="Products-10K",
        name_zh="Products-10K",
        category="ecommerce",
        category_zh="电商版权保护",
        description="Products-10K (JD AI Research) is a SKU-level product recognition dataset from JD.com e-commerce catalog.",
        description_zh="Products-10K（京东 AI 研究院）是京东电商 SKU 级商品识别数据集。",
        source_url="https://products-10k.github.io/",
        official_total_images=190_000,
    ),
    DatasetCatalogEntry(
        id="rico",
        name="RICO",
        name_zh="RICO",
        category="mobile-ui",
        category_zh="移动端截图和 UI 内容保护",
        description="RICO (Google) collects Android app screenshots with view hierarchies and UI element annotations.",
        description_zh="RICO（Google）收录 Android 应用截图及视图层级与 UI 元素标注。",
        source_url="http://www.interactionmining.org/rico",
        official_total_images=66_000,
    ),
    DatasetCatalogEntry(
        id="mobileviews",
        name="MobileViews",
        name_zh="MobileViews",
        category="mobile-ui",
        category_zh="移动端截图和 UI 内容保护",
        description="MobileViews is a large-scale Android GUI dataset with screenshot–view-hierarchy pairs for mobile agents.",
        description_zh="MobileViews 是大规模 Android GUI 截图与视图层级配对数据集，面向移动端智能体研究。",
        source_url="https://huggingface.co/datasets/mllmTeam/MobileViews",
        official_total_images=1_200_000,
    ),
)


def get_catalog_entry(dataset_id: str) -> DatasetCatalogEntry:
    for entry in DATASET_CATALOG:
        if entry.id == dataset_id:
            return entry
    raise KeyError(f"Unknown dataset catalog id: {dataset_id}")


def dataset_root(resources_root: Path, dataset_id: str) -> Path:
    return resources_root / "datasets" / dataset_id


def compact_dir(resources_root: Path, dataset_id: str, *, compact_uses_root: bool = False) -> Path:
    root = dataset_root(resources_root, dataset_id)
    compact = root / "compact"
    if compact.exists() and any(compact.iterdir()):
        return compact
    if compact_uses_root and root.exists():
        return root
    return compact


def full_dir(resources_root: Path, dataset_id: str) -> Path:
    root = dataset_root(resources_root, dataset_id)
    full = root / "full"
    if full.exists() and any(full.iterdir()):
        return full
    return root


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return len(iter_image_paths(path))


def custom_pool_count(resources_root: Path, entry: DatasetCatalogEntry) -> int:
    root = dataset_root(resources_root, entry.id)
    full_path = root / "full"
    if full_path.exists() and any(full_path.iterdir()):
        return count_images(full_path)
    compact_path = compact_dir(resources_root, entry.id, compact_uses_root=entry.compact_uses_root)
    compact_images = {path.resolve() for path in iter_image_paths(compact_path)}
    root_images = iter_image_paths(root)
    pool = [path for path in root_images if path.resolve() not in compact_images]
    if pool:
        return len(pool)
    if root_images:
        return len(root_images)
    return entry.official_total_images or 0


def resolve_local_paths(resources_root: Path, entry: DatasetCatalogEntry) -> dict[str, Any]:
    root = dataset_root(resources_root, entry.id)
    compact_path = compact_dir(resources_root, entry.id, compact_uses_root=entry.compact_uses_root)
    full_path = full_dir(resources_root, entry.id)
    compact_count = count_images(compact_path)
    full_count = count_images(full_path)
    local_root_count = count_images(root) if root.exists() else 0
    pool_count = custom_pool_count(resources_root, entry)

    compact_available = compact_count > 0
    local_available = full_count > 0 or local_root_count > 0

    return {
        "rootPath": str(root),
        "compactPath": str(compact_path),
        "fullPath": str(full_path),
        "compactSampleCount": compact_count,
        "fullSampleCount": max(full_count, local_root_count),
        "customPoolCount": pool_count,
        "compactAvailable": compact_available,
        "localAvailable": local_available,
        "installed": compact_available or local_available,
    }


def build_catalog_item(resources_root: Path, entry: DatasetCatalogEntry) -> dict[str, Any]:
    local = resolve_local_paths(resources_root, entry)
    custom_ready = entry.manifest_url is not None or local["customPoolCount"] > 0
    return {
        **entry.to_json(),
        **local,
        "customDownloadReady": custom_ready,
        "remoteManifestConfigured": entry.manifest_url is not None,
    }


def list_dataset_catalog(resources_root: Path) -> list[dict[str, Any]]:
    items = [build_catalog_item(resources_root, entry) for entry in DATASET_CATALOG]
    scanned_ids = {item["id"] for item in items}

    datasets_root = resources_root / "datasets"
    if datasets_root.exists():
        for child in sorted(datasets_root.iterdir()):
            if not child.is_dir() or child.name in scanned_ids:
                continue
            images = iter_image_paths(child)
            if not images:
                continue
            count = len(images)
            items.append(
                {
                    "id": child.name,
                    "name": child.name.replace("_", " ").replace("-", " ").title(),
                    "nameZh": child.name.replace("_", " ").replace("-", " ").title(),
                    "category": "local",
                    "categoryZh": "本地数据集",
                    "description": "Locally discovered dataset folder.",
                    "descriptionZh": "本地扫描发现的数据集目录。",
                    "sourceUrl": "",
                    "officialTotalImages": count,
                    "manifestUrl": None,
                    "compactSampleCount": count,
                    "compactUsesRoot": True,
                    "rootPath": str(child),
                    "compactPath": str(child),
                    "fullPath": str(child),
                    "fullSampleCount": count,
                    "customPoolCount": count,
                    "compactAvailable": True,
                    "localAvailable": True,
                    "installed": True,
                    "customDownloadReady": True,
                    "remoteManifestConfigured": False,
                }
            )
    return items


def list_categories() -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for entry in DATASET_CATALOG:
        seen[entry.category] = entry.category_zh
    seen["local"] = "本地数据集"
    return [{"id": key, "nameZh": value} for key, value in seen.items()]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS
