import cv2
import numpy as np

from geosunpre.experiments.refinement import refine_native_corners
from geosunpre.statistics import paired_track_bootstrap


def test_native_corner_refinement_keeps_corner_coordinates_and_rejects_edges():
    yy,xx=np.mgrid[:96,:96]
    gray=np.where((xx<48)==(yy<48),230,25).astype(np.uint8)
    gray=cv2.GaussianBlur(gray,(5,5),.7)
    # The physical corner is (48,48) in COLMAP coordinates, (47.5,47.5) in OpenCV.
    pixels=np.array([[49.1,46.8],[48.,20.],[20.,20.]],dtype=np.float32)
    refined,accepted,_=refine_native_corners(gray,pixels)
    assert accepted.tolist()==[True,False,False]
    np.testing.assert_allclose(refined[0],[48.,48.],atol=.03)


def test_paired_track_bootstrap_records_its_finite_subset_without_inf_subtraction():
    tracks=np.array(['a','a','b','c','d'])
    baseline={'track_id':tracks,'error':np.array([2.,3.,4.,5.,np.inf])}
    candidate={'track_id':tracks,'error':np.array([1.5,2.5,3.5,4.5,np.inf])}
    with np.errstate(invalid='raise'):
        result=paired_track_bootstrap({'pro_initial':baseline,'candidate':candidate},
                                      np.ones(5,dtype=bool),replicates=100)['candidate']
    assert result['queries']==4 and result['tracks']==3
    assert result['median_delta_px']==-.5
    assert result['interval95_px']==[-.5,-.5]
