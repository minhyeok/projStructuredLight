
// Define structure for storing structured lighting parameters.
struct slParams{

	// Output options.
	char outdir[1024];	            // base output directory
	char object[1024];              // object name
	bool save;                      // enable/disable saving of image sequence

	// Camera options.
	int  cam_w;                     // camera columns
	int  cam_h;                     // camera rows

	// Projector options.
	int  proj_w;                    // projector columns
	int  proj_h;                    // projector rows

	// Projector-camera gain parameters.
	int cam_gain;                   // scale factor for camera images
	int proj_gain;                  // scale factor for projector images

	// Calibration model options.
	bool cam_dist_model[2];         // enable/disable [tangential, 6th-order radial] distortion components for camera
	bool proj_dist_model[2];        // enable/disable [tangential, 6th-order radial] distortion components for projector

	// Define camera calibration chessboard parameters.
	// Note: Width/height are number of "interior" corners, excluding outside edges.
	int   cam_board_w;              // interior chessboard corners (along width)
	int   cam_board_h;              // interior chessboard corners (along height)
	float cam_board_w_mm;           // physical length of chessboard square (width in mm)
	float cam_board_h_mm;           // physical length of chessboard square (height in mm)

	// Define projector calibration chessboard parameters.
	// Note: Width/height are number of "interior" corners, excluding outside edges.
	int proj_board_w;               // interior chessboard corners (along width)
	int proj_board_h;               // interior chessboard corners (along height)
	int proj_board_w_pixels;        // physical length of chessboard square (width in pixels)
	int proj_board_h_pixels;        // physical length of chessboard square (height in pixels)

	// General options.

	int   delay;                    // frame delay between projection and image capture (in ms)
	int   thresh;                   // minimum contrast threshold for decoding (maximum of 255)
	float dist_range[2];            // {minimum, maximum} distance (from camera), otherwise point is rejected
	float dist_reject;              // rejection distance (for outlier removal) if row and column scanning are both enabled (in mm)
	float background_depth_thresh;  // threshold distance for background removal (in mm)	

	// Visualization options.
	bool display;                   // enable/disable display of intermediate results (e.g., image sequence, calibration data, etc.)
	int window_w;                   // camera display window width (height is derived)
	int window_h;                   // camera display window width (derived parameter)
};

// Define structure for structured lighting calibration parameters.
struct slCalib{

	// Camera calibration.
	CvMat* cam_intrinsic;           // camera intrinsic parameter matrix
	CvMat* cam_distortion;          // camera distortion coefficient vector
	CvMat* cam_extrinsic;           // camera extrinsic parameter matrix

	// Projector calibration.
	CvMat* proj_intrinsic;          // projector intrinsic parameter matrix
	CvMat* proj_distortion;         // projector distortion coefficient vector
	CvMat* proj_extrinsic;          // projector extrinsic parameter matrix

	// Projector-camera geometric parameters.
	// Note: All quantities defined in the camera coordinate system.
	CvMat* cam_center;              // camera center of projection
	CvMat* proj_center;             // projector center of projection
	CvMat* cam_rays;                // optical rays for each camera pixel
	CvMat* proj_rays;               // optical rays for each projector pixel
	CvMat* proj_planes[8]; 

	// Flags to indicate calibration status.
	bool cam_intrinsic_calib;       // flag to indicate state of intrinsic camera calibration
    bool proj_intrinsic_calib;		// flag to indicate state of intrinsic projector calibration
	bool procam_extrinsic_calib;    // flag to indicate state of extrinsic projector-camera calibration

	// Background model (used to segment foreground objects of interest from static background).
	CvMat*    background_depth_map; // background depth map
	IplImage* background_image;     // background image 
	IplImage* background_mask;      // background mask
};