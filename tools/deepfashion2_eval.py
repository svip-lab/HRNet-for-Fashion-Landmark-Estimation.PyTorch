
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pylab
import sys


assert len(sys.argv) == 3, f'usage: {sys.argv[0]} <annotation json> <result json>'
annFile = sys.argv[1]
resFile = sys.argv[2]

cocoGt=COCO(annFile)
imgIds=sorted(cocoGt.getImgIds())
cocoDt=cocoGt.loadRes(resFile)

# # evaluating clothes detection
# cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
# cocoEval.params.imgIds  = imgIds
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()

# evaluating landmark and pose Estimation
cocoEval = COCOeval(cocoGt,cocoDt,'keypoints')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# # evaluating clothes segmentation
# cocoEval = COCOeval(cocoGt,cocoDt,'segm')
# cocoEval.params.imgIds  = imgIds
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()