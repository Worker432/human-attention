"""Object shares of sampled model points, never measured dwell time."""
from collections import Counter
from scanpath_prediction import assign_steps

HEATMAP_VARIANTS = 30


def object_attention(predictions):
    if not predictions:
        return dict(data_source='model_prediction', variants=0, total_points=0, rows=[])
    first = predictions[0]
    for result in predictions:
        if (result['image_sha256'], result['image_width'], result['image_height']) != (
                first['image_sha256'], first['image_width'], first['image_height']):
            raise ValueError('Нельзя объединять прогнозы разных изображений')
    boxes = first['objects']
    points = [(p['x'], p['y']) for r in predictions for p in r['fixations']]
    mapped = assign_steps(points, boxes)
    counts = Counter((p['assignment'], p['object_id']) for p in mapped)
    rows = []
    for box in boxes:
        rows.append(dict(assignment='assigned', object_id=box['object_id'],
                         class_name=box['class_name'], xyxy=box['xyxy'],
                         count=counts['assigned', box['object_id']]))
    rows.sort(key=lambda row: (-row['count'], row['object_id']))
    for assignment in ('ambiguous', 'outside'):
        rows.append(dict(assignment=assignment, object_id=None, class_name=None,
                         xyxy=None, count=counts[assignment, None]))
    for row in rows:
        row['share'] = row['count']/len(points) if points else 0.
    return dict(data_source='model_prediction', metric='share_of_sampled_points',
                image_sha256=first['image_sha256'], variants=len(predictions),
                seeds=[r['parameters']['seed'] for r in predictions],
                total_points=len(points), rows=rows,
                note='Shares of predicted points, not fixation duration or percentage of people. '
                     'All points are reassigned to the first variant\'s detection boxes; '
                     'overlaps and outside points are separate exclusive categories.')
