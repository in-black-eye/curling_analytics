import os
import glob
from typing import List, Tuple, Dict
import xml.etree.ElementTree as ET

import cv2


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(root: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(root, e)))
    files.sort()
    return files


def read_image_size(path: str) -> Tuple[int, int]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    return w, h


def parse_cvat_annotations(xml_path: str) -> Tuple[Dict[str, List[Tuple[int, float, float]]], List[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    names: List[str] = []
    label_to_id: Dict[str, int] = {}
    meta = root.find('meta')
    if meta is not None:
        task = meta.find('task')
        if task is not None:
            labels_el = task.find('labels')
            if labels_el is not None:
                for lab in labels_el.findall('label'):
                    nm_el = lab.find('name')
                    if nm_el is not None:
                        nm = (nm_el.text or '').strip()
                        if nm and nm not in label_to_id:
                            label_to_id[nm] = len(names)
                            names.append(nm)

    mapping: Dict[str, List[Tuple[int, float, float]]] = {}
    for img_el in root.findall('image'):
        img_name = img_el.get('name')
        if not img_name:
            continue
        acc: List[Tuple[int, float, float]] = []
        # points at image level
        for p in img_el.findall('points'):
            lbl = p.get('label') or ''
            if lbl not in label_to_id:
                label_to_id[lbl] = len(names)
                names.append(lbl)
            cls_id = label_to_id[lbl]
            pts = (p.get('points') or '').split(';')
            for pair in pts:
                pair = pair.strip()
                if not pair:
                    continue
                try:
                    xs, ys = pair.split(',')
                    acc.append((cls_id, float(xs), float(ys)))
                except Exception:
                    continue
        # points inside skeletons
        for sk in img_el.findall('skeleton'):
            for p in sk.findall('points'):
                lbl = p.get('label') or ''
                if lbl not in label_to_id:
                    label_to_id[lbl] = len(names)
                    names.append(lbl)
                cls_id = label_to_id[lbl]
                pts = (p.get('points') or '').split(';')
                for pair in pts:
                    pair = pair.strip()
                    if not pair:
                        continue
                    try:
                        xs, ys = pair.split(',')
                        acc.append((cls_id, float(xs), float(ys)))
                    except Exception:
                        continue
        if acc:
            mapping[os.path.basename(img_name)] = acc
    if not names:
        names = [f"kpt_{i}" for i in range(22)]
    return mapping, names


def write_boxes_from_pixel_points(
    image_path: str,
    out_label_path: str,
    points_with_cls: List[Tuple[int, float, float]],
    box_size_px: int,
    original_size: Tuple[int, int] | None = None
) -> int:
    ensure_dir(os.path.dirname(out_label_path))
    w, h = read_image_size(image_path)
    half = box_size_px / 2.0
    entries: List[str] = []
    for cls_id, px, py in points_with_cls:
        cx, cy = px, py
        if original_size is not None:
            orig_w, orig_h = original_size
            if orig_w > 0 and orig_h > 0:
                scale_x = w / float(orig_w)
                scale_y = h / float(orig_h)
                cx *= scale_x
                cy *= scale_y
        cx = min(max(cx, 0.0), w - 1.0)
        cy = min(max(cy, 0.0), h - 1.0)
        x1 = max(0.0, cx - half)
        y1 = max(0.0, cy - half)
        x2 = min(w - 1.0, cx + half)
        y2 = min(h - 1.0, cy + half)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        ycx = (x1 + x2) / 2.0 / w
        ycy = (y1 + y2) / 2.0 / h
        yw = bw / w
        yh = bh / h
        entries.append(f"{cls_id} {ycx:.6f} {ycy:.6f} {yw:.6f} {yh:.6f}")
    with open(out_label_path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(e + '\n')
    return len(entries)


def build_detection_dataset(
    dataset_root: str,
    output_dataset_root: str,
    yaml_path: str,
    box_size_px: int = 25,
    resize_max_width: int | None = 1280,
):
    # Find annotations.xml
    annotations_xml = None
    for cand in [os.path.join(dataset_root, 'annotations.xml'),
                 os.path.join(dataset_root, 'annotations', 'annotations.xml')]:
        if os.path.isfile(cand):
            annotations_xml = cand
            break
    if not annotations_xml:
        raise FileNotFoundError(f"annotations.xml not found under {dataset_root}")

    mapping, names = parse_cvat_annotations(annotations_xml)

    # Map image name to full path
    all_img_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        all_img_files.extend(glob.glob(os.path.join(dataset_root, '**', ext), recursive=True))
    name_to_path: Dict[str, str] = {os.path.basename(p): p for p in all_img_files}

    image_names = list(mapping.keys())
    # split 90/10
    n = len(image_names)
    val_n = max(1, int(round(n * 0.1)))
    train_names, val_names = image_names[val_n:], image_names[:val_n]

    out_train_img = os.path.join(output_dataset_root, 'images', 'train')
    out_val_img = os.path.join(output_dataset_root, 'images', 'val')
    out_train_lbl = os.path.join(output_dataset_root, 'labels', 'train')
    out_val_lbl = os.path.join(output_dataset_root, 'labels', 'val')
    ensure_dir(out_train_img); ensure_dir(out_val_img)
    ensure_dir(out_train_lbl); ensure_dir(out_val_lbl)

    def process_split(names_list: List[str], out_img_dir: str, out_lbl_dir: str):
        for nm in names_list:
            if nm not in name_to_path:
                continue
            src_img_path = name_to_path[nm]
            dst_img_path = os.path.join(out_img_dir, os.path.basename(nm))
            im = cv2.imread(src_img_path)
            if im is None:
                continue
            orig_h, orig_w = im.shape[:2]
            if resize_max_width is not None and orig_w > resize_max_width:
                scale = resize_max_width / float(orig_w)
                new_w = resize_max_width
                new_h = int(round(orig_h * scale))
                im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_img_path, im)
            pts = mapping.get(nm, [])
            dst_lbl = os.path.join(out_lbl_dir, os.path.splitext(os.path.basename(nm))[0] + '.txt')
            _ = write_boxes_from_pixel_points(
                dst_img_path, dst_lbl, pts, box_size_px=box_size_px, original_size=(orig_w, orig_h)
            )

    process_split(train_names, out_train_img, out_train_lbl)
    process_split(val_names, out_val_img, out_val_lbl)

    # Write YAML
    ensure_dir(os.path.dirname(yaml_path))
    yaml_content = (
        f"path: \"{output_dataset_root}\"\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"Dataset built at: {output_dataset_root} (box={box_size_px}px)")


