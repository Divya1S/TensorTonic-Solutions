def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size
    anchors = []
    
    # Iterate over grid cells in row-major order (i then j)
    for i in range(feature_size):
        for j in range(feature_size):
            # Compute center in image coordinates
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            
            # Iterate over scales and aspect ratios
            for s in scales:
                for r in aspect_ratios:
                    # Compute width and height
                    w = s * math.sqrt(r)
                    h = s / math.sqrt(r)
                    
                    # Compute box coordinates
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchors.append([x1, y1, x2, y2])
                    
    return anchors