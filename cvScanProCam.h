
// Run the background capture (used to eliminate background points from reconstructions).
int runBackgroundCapture(CvCapture* capture, struct slParams* sl_params, struct slCalib* sl_calib,int Angle);

// Run the structured light scanner.
int runStructuredLight(CvCapture* capture, struct slParams* sl_params, struct slCalib* sl_calib,int scan_index,int Angle);
		
int Canny(CvCapture* capture, IplImage* cam_image, struct slParams* sl_params, IplImage* edge_image,IplImage* binary_image);

int All_runStructuredLight(CvCapture* capture, struct slParams* sl_params,struct slCalib* sl_calib,int scan_index);

int depthMapCompare(CvMat * src1, CvMat* src2, IplImage * gray_mask, struct slParams * sl_params, int num1, int num2);