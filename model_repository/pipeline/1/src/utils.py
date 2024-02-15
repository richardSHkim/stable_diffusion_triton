DATA_NAME_LIST = [
    "levatas",
    "nuImages",
    "roboflow-cable-damage",
    "roboflow-pcb-tiled",
]


def create_reco_prompt(
    caption: str = '',
    phrases=[],
    boxes=[],
    normalize_boxes=True,
    w=512,
    h=512,
    num_bins=1000,
    ):
    """
    method to create ReCo prompt

    caption: global caption
    phrases: list of regional captions
    boxes: list of regional coordinates (unnormalized xyxy)
    """

    SOS_token = '<|startoftext|>'
    EOS_token = '<|endoftext|>'
    
    box_captions_with_coords = []
    
    box_captions_with_coords += [caption]
    box_captions_with_coords += [EOS_token]

    for phrase, box in zip(phrases, boxes):
        if normalize_boxes:
            x1, y1, x2, y2 = box
            box = [x1/w, y1/h, x2/w, y2/h]

        # quantize into bins
        quant_x0 = int(round((box[0] * (num_bins - 1))))
        quant_y0 = int(round((box[1] * (num_bins - 1))))
        quant_x1 = int(round((box[2] * (num_bins - 1))))
        quant_y1 = int(round((box[3] * (num_bins - 1))))
        
        # ReCo format
        # Add SOS/EOS before/after regional captions
        box_captions_with_coords += [
            f"<bin{str(quant_x0).zfill(3)}>",
            f"<bin{str(quant_y0).zfill(3)}>",
            f"<bin{str(quant_x1).zfill(3)}>",
            f"<bin{str(quant_y1).zfill(3)}>",
            SOS_token,
            phrase,
            EOS_token
        ]

    text = " ".join(box_captions_with_coords)
    return text