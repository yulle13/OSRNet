import networkx as nx
import numpy as np
import cv2
import re


our_class = {
    "Plastic bottle": 1,
    "Plastic box": 2,
    "Paper box": 3,
    "Paper cup": 4,
    "Metal bottle": 5,
    "Metal box": 6,
    "Glass bottle": 7,
    "Glass cup": 8,
    'Plastic cup': 9,
    "Wooden": 10,
    "Cloth": 11,
    "Shoes": 12,
    "Book": 13
}


def draw_graph(matrix, ind=None, pos=None):
    edges = np.where(matrix == 1)
    if ind is not None:
        from_idx = []
        to_idx = []
        for i in range(len(edges[0])):
            if edges[0][i] in ind and edges[1][i] in ind:
                from_idx.append(edges[0][i])
                to_idx.append(edges[1][i])
    else:
        from_idx = edges[0].tolist()
        to_idx = edges[1].tolist()

    from_node = [str(i) for i in from_idx]
    to_node = [str(i) for i in to_idx]
    nodes = [str(i) for i in range(matrix.shape[0])]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(list(zip(from_node, to_node)))
    if pos is None:
        #         pos = nx.kamada_kawai_layout(G)
        pos = nx.spring_layout(G, k=0.5, iterations=20)
    #         pos = nx.planar_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos, font_color='w')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2)
    return pos


def get_modal(label, h, w):
    points = label['points']
    _label = label['label']
    # 将points转成seg, 得到对于的modal
    poly = np.array(points).reshape(-1, 2)
    segmentation = [poly.flatten().tolist()]
    mask = np.zeros((h, w), dtype=np.uint8)
    seg = np.array(segmentation, dtype=np.int32)
    seg = seg.reshape((-1, 1, 2))
    # category = find_category(_label)
    # if category == 0:
    #     print(_label)
    modal = cv2.fillPoly(mask, [seg], 1)
    return modal


def mask_to_bbox(mask):
    mask = (mask != 0)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()] # xywh


def find_category(label):
    _class = re.search(r'_(.*?)_', label)
    if _class.group(1) in our_class:
        category = our_class[_class.group(1)]
    else:
        print('标签类别错误')
        category = 0
    return category


def bbox_iou(b1, b2):
    '''
    b: (x1,y1,x2,y2)
    '''
    lx = max(b1[0], b2[0])
    rx = min(b1[2], b2[2])
    uy = max(b1[1], b2[1])
    dy = min(b1[3], b2[3])
    if rx <= lx or dy <= uy:
        return 0.
    else:
        interArea = (rx-lx)*(dy-uy)
        a1 = float((b1[2] - b1[0]) * (b1[3] - b1[1]))
        a2 = float((b2[2] - b2[0]) * (b2[3] - b2[1]))
        return interArea / (a1 + a2 - interArea)


def crop_padding(img, roi, pad_value):
    '''
    img: HxW or HxWxC np.ndarray
    roi: (x,y,w,h)
    pad_value: [b,g,r]
    '''
    need_squeeze = False
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
        need_squeeze = True
    assert len(pad_value) == img.shape[2]
    x,y,w,h = roi
    x,y,w,h = int(x),int(y),int(w),int(h)
    H, W = img.shape[:2]
    output = np.tile(np.array(pad_value), (h, w, 1)).astype(img.dtype)
    if bbox_iou((x,y,x+w,y+h), (0,0,W,H)) > 0:
        output[max(-y,0):min(H-y,h), max(-x,0):min(W-x,w), :] = img[max(y,0):min(y+h,H), max(x,0):min(x+w,W), :]
    if need_squeeze:
        output = np.squeeze(output)
    return output
