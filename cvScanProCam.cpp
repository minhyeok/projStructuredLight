#include "stdafx.h"
#include "cvStructuredLight.h"
#include "cvScanProCam.h"
#include "cvUtilProCam.h"
#include <math.h>
#include <iostream>   
#include <algorithm>   
using namespace std;   

// Generate Gray codes.
int generateGrayCodes(int width, int height, 
	IplImage**& gray_codes,
	int& p_num,
	int& log_num,
	int& n_shift, 
	int Angle){
		// Determine number of required codes and row/column offsets.
		int row,col,distant,direct;
		switch(Angle){
		case 0		:	{col=1;row=0;distant=0;direct=1;} break;
		case 26		:	{col=0;row=1;distant=2;direct=1;} break;
		case 45		:	{col=1;row=0;distant=1;direct=1;} break;
		case 63		:	{col=1;row=0;distant=2;direct=1;} break;
		case 90		:	{col=0;row=1;distant=0;direct=1;} break;
		case 116	:	{col=1;row=0;distant=2;direct=-1;} break;
		case 135	:	{col=1;row=0;distant=1;direct=-1;} break;
		case 154	:	{col=0;row=1;distant=2;direct=-1;} break;
		default		:	break;
		}


		p_num = (width*col+height*row)*distant + (height*col+width*row-distant);
		log_num = (int)ceil(log2(p_num));
		n_shift = (int)floor((pow(2.0,log_num)-p_num)/2);

		// Allocate Gray codes.
		gray_codes = new IplImage* [log_num+1];
		for(int i=0; i<(log_num+1); i++)
			gray_codes[i] = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U, 1);
		int step = gray_codes[0]->widthStep/sizeof(uchar);

		// Define first code as a white image.
		cvSet(gray_codes[0], cvScalar(255));

		if(Angle==0){
			for(int r=0; r<height; r++){
				for(int i=0; i<log_num; i++){
					uchar* data = (uchar*)gray_codes[i+1]->imageData;
					if(i>0)
						data[r*step] = (((r+n_shift) >> (log_num-i-1)) & 1)^(((r+n_shift) >> (log_num-i)) & 1);
					else
						data[r*step] = (((r+n_shift) >> (log_num-i-1)) & 1);
					data[r*step] *= 255;
					for(int c=1; c<width; c++)
						data[r*step+c] = data[r*step];	
				}
			}
		}
		else if(Angle==90){
			for(int c=0; c<width; c++){
				for(int i=0; i<log_num; i++){
					uchar* data = (uchar*)gray_codes[i+1]->imageData;
					if(i>0)
						data[c] = (((c+n_shift) >> (log_num-i-1)) & 1)^(((c+n_shift) >> (log_num-i)) & 1);
					else
						data[c] = (((c+n_shift) >> (log_num-i-1)) & 1);
					data[c] *= 255;
					for(int r=1; r<height; r++)
						data[r*step+c] = data[c];	
				}
			}
		}
		else{
			for(int c=0; c<((width+height)*(col+row)-distant); c++){
				for(int i=0; i<log_num; i++){
					uchar* data = (uchar*)gray_codes[i+1]->imageData;
					if(0<direct){
						if(i>0){
							if(c<width*col+height*row)
								for(int k =0;k<distant;k++)
									data[(c*row+k*col)*step+(c*col+k*row)] = 
									(((((c*distant+k)+n_shift) >> (log_num-i-1)) & 1)^((((c*distant+k)+n_shift) >> (log_num-i)) & 1))*255;
							else
								data[((height-1)*row+(distant+c-width)*col)*step+((width-1)*col+(distant+c-height)*row)] = 
								(((((c+(width*col+height*row)*(distant-1))+n_shift) >> (log_num-i-1)) & 1)^((((c+(width*col+height*row)*(distant-1))+n_shift) >> (log_num-i)) & 1))*255;
						}

						else{
							if(c<width*col+height*row)
								for(int k =0;k<distant;k++)
									data[(c*row+k*col)*step+(c*col+k*row)] = ((((c*distant+k)+n_shift) >> (log_num-i-1)) & 1)*255;
							else
								data[((height-1)*row+(distant+c-width)*col)*step+((width-1)*col+(distant+c-height)*row)] = 
								((((c+(width*col+height*row)*(distant-1))+n_shift) >> (log_num-i-1)) & 1)*255;
						}
					}
					else{
						if(i>0){
							if(c<col*(height-distant)+height*row)
								for(int k=0;k<(distant*row+col);k++)
									data[((height-c-1)*col+c*row)*step+(width-1-k)*row] = 
									(((((c*col+(c*distant+k)*row)+n_shift) >> (log_num-i-1)) & 1)^((((c*col+(c*distant+k)*row)+n_shift) >> (log_num-i)) & 1))*255;
							else
								for(int k=0;k<(distant*col+row);k++){
									int num = ((c-height+distant)*(distant-1)+c+k)*col + (c+height*(distant-1))*row;
									data[((distant-1-k)*col+(height-1)*row)*step+(c-height+distant)*col+(width-distant-1-c+height)*row] =
										((((num+n_shift) >> (log_num-i-1)) & 1)^(((num+n_shift) >> (log_num-i)) & 1))*255;
								}
						}
						else{
							if(c<col*(height-distant)+height*row)
								for(int k=0;k<(distant*row+col);k++){
									int num_2 = (c*distant+k)*row+c*col;
									data[((height-c-1)*col+c*row)*step+(width-1-k)*row] = (((num_2+n_shift) >> (log_num-i-1)) & 1)*255;
								}

							else
								for(int k=0;k<(distant*col+row);k++){
									int num_3 = ((c-height+distant)*(distant-1)+c+k)*col+(c+height*(distant-1))*row;
									data[((distant-1-k)*col+(height-1)*row)*step+(c-height+distant)*(col-row)+(width-1)*row] =
										(((num_3+n_shift) >> (log_num-i-1)) & 1)*255;
								}
						}
					}
				}
			}


			int real = (height*col+width*row)/distant;
			int mod = (height*col+width*row) - real*distant;
			for(int i=0; i<log_num; i++){
				uchar* data = (uchar*)gray_codes[i+1]->imageData;
				for(int r=0; r<real+mod-1; r++){
					for(int c=0; c<((width*col+height*row)-1); c++){
						if(0<direct){
							if(r<real-1)
								for(int r_s = 0; r_s<distant; r_s++)
									data[(((r+1)*distant+r_s)*col+c*row)*step+(c*col+(r_s+(r+1)*distant)*row)] = 
									data[((r*distant+r_s)*col+(c+1)*row)*step+((c+1)*col+(r_s+r*distant)*row)];
							else
								for(int r_s = 0; r_s<mod; r_s++)
									data[(((r+1)*distant+r_s)*col+c*row)*step+(c*col+(r_s+(r+1)*distant)*row)] = 
									data[((r*distant+r_s)*col+(c+1)*row)*step+((c+1)*col+(r_s+r*distant)*row)];
						}
						else{
							if(r<real-1)
								for(int r_s = 0; r_s<distant; r_s++)
									data[(((r+1)*distant+r_s)*col+c*row)*step+((c+1)*col+(-r_s+width-1-(r+1)*distant)*row)] = 
									data[((r*distant+r_s)*col+(c+1)*row)*step+(c*col+(-r_s+width-1-r*distant)*row)];
							else
								for(int r_s = 0; r_s<mod; r_s++)
									data[(((r+1)*distant+r_s)*col+c*row)*step+((c+1)*col+(-r_s+width-1-(r+1)*distant)*row)] = 
									data[((r*distant+r_s)*col+(c+1)*row)*step+(c*col+(-r_s+width-1-r*distant)*row)];
						}
					}
				}
			}
		}

		return 0;
}

// Decode Gray codes.
int decodeGrayCodes(int proj_width, int proj_height,
	IplImage**& gray_codes,
	IplImage*& decoded_image,
	IplImage*& mask,
	int& n_plane,int& r_plane, int& n_shift, 
	int sl_thresh){

		// Extract width and height of images.
		int cam_width  = gray_codes[0]->width;
		int cam_height = gray_codes[0]->height;

		// Allocate temporary variables.
		IplImage* gray_1      = cvCreateImage(cvSize(cam_width, cam_height), IPL_DEPTH_8U,  1);
		IplImage* gray_2      = cvCreateImage(cvSize(cam_width, cam_height), IPL_DEPTH_8U,  1);
		IplImage* bit_plane_1 = cvCreateImage(cvSize(cam_width, cam_height), IPL_DEPTH_8U,  1);
		IplImage* bit_plane_2 = cvCreateImage(cvSize(cam_width, cam_height), IPL_DEPTH_8U,  1);
		IplImage* temp        = cvCreateImage(cvSize(cam_width, cam_height), IPL_DEPTH_8U,  1);

		// Initialize image mask (indicates reconstructed pixels).
		cvSet(mask, cvScalar(0));
		// Decode Gray codes for projector diagonals.

		cvZero(decoded_image);
		for(int i=0; i<n_plane; i++){

			// Decode bit-plane and update mask.
			cvCvtColor(gray_codes[2*(i+1)],   gray_1, CV_RGB2GRAY);
			cvCvtColor(gray_codes[2*(i+1)+1], gray_2, CV_RGB2GRAY);
			cvAbsDiff(gray_1, gray_2, temp);
			cvCmpS(temp, sl_thresh, temp, CV_CMP_GE);
			cvOr(temp, mask, mask);
			cvCmp(gray_1, gray_2, bit_plane_2, CV_CMP_GE);

			// Convert from gray code to decimal value.
			if(i>0)
				cvXor(bit_plane_1, bit_plane_2, bit_plane_1);
			else
				cvCopyImage(bit_plane_2, bit_plane_1);
			cvAddS(decoded_image, cvScalar(pow(2.0,n_plane-i-1)), decoded_image, bit_plane_1);
		}
		cvSubS(decoded_image, cvScalar(n_shift), decoded_image);




		// Eliminate invalid column/row estimates.
		// Note: This will exclude pixels if either the column or row is missing or erroneous.
		cvCmpS(decoded_image, r_plane,  temp, CV_CMP_LE);
		cvAnd(temp, mask, mask);
		cvCmpS(decoded_image, 0,  temp, CV_CMP_GE);
		cvAnd(temp, mask, mask);
		cvNot(mask, temp);
		cvSet(decoded_image, cvScalar(NULL), temp);

		// Free allocated resources.
		cvReleaseImage(&gray_1);
		cvReleaseImage(&gray_2);
		cvReleaseImage(&bit_plane_1);
		cvReleaseImage(&bit_plane_2);
		cvReleaseImage(&temp);

		// Return without errors.
		return 0;
}






// Illuminate object with a structured light sequence.
int slScan(CvCapture* capture, 
	IplImage**& proj_codes, IplImage**& cam_codes, int n_plane, 
struct slParams* sl_params,
struct slCalib*  sl_calib){

	// Create a window to display captured frames.
	IplImage* cam_frame  = cvQueryFrame2(capture, sl_params);
	IplImage* proj_frame = cvCreateImage(cvSize(sl_params->proj_w, sl_params->proj_h), IPL_DEPTH_8U, 1);
	cvNamedWindow("camWindow", CV_WINDOW_AUTOSIZE);

	cvWaitKey(1);

	// Allocate storage for captured images.
	cam_codes = new IplImage* [2*(n_plane+1)];
	for(int i=0; i<2*(n_plane+1); i++)
		cam_codes[i] = cvCloneImage(cam_frame);

	// Capture structured light sequence.
	// Note: Assumes sequence is binary, so code and its inverse can be compared.
	IplImage* inverse_code = cvCloneImage(proj_codes[0]);


	for(int i=0; i<(n_plane+1); i++){

		// Display code.
		cvCopy(proj_codes[i], proj_frame);
		cvScale(proj_frame, proj_frame, 2.*(sl_params->proj_gain/100.), 0);
		cvShowImage("projWindow", proj_frame);
		cvWaitKey(sl_params->delay);
		cam_frame = cvQueryFrameSafe(capture, sl_params, false);
		cvScale(cam_frame, cam_frame, 2.*(sl_params->cam_gain/100.), 0);
		cvShowImageResampled("camWindow", cam_frame, sl_params->window_w, sl_params->window_h);
		cvCopyImage(cam_frame, cam_codes[2*i]);

		// Display inverse code.
		cvSubRS(proj_codes[i], cvScalar(255), inverse_code);
		cvCopy(inverse_code, proj_frame);
		cvScale(proj_frame, proj_frame, 2.*(sl_params->proj_gain/100.), 0);
		cvShowImage("projWindow", proj_frame);
		cvWaitKey(sl_params->delay);
		cam_frame = cvQueryFrameSafe(capture, sl_params, false);
		cvScale(cam_frame, cam_frame, 2.*(sl_params->cam_gain/100.), 0);
		cvShowImageResampled("camWindow", cam_frame, sl_params->window_w, sl_params->window_h);
		cvCopyImage(cam_frame, cam_codes[2*i+1]);
	}

	// Display black projector image.
	cvSet(inverse_code, cvScalar(0, 0, 0));
	cvShowImage("projWindow", inverse_code);
	cvWaitKey(1);

	// Free allocated resources.
	cvReleaseImage(&inverse_code);
	cvReleaseImage(&proj_frame);

	// Return without errors.
	cvDestroyWindow("camWindow");
	return 0;
}

// Display a structured lighting decoding result (i.e., projector column/row to camera pixel correspondences).
int displayDecodingResults(IplImage*& decoded_image,
	IplImage*& mask,
struct slParams* sl_params){

	// Create a window to display correspondences.
	cvNamedWindow("camWindow", CV_WINDOW_AUTOSIZE);

	cvWaitKey(1);

	// Allocate image arrays.
	IplImage* temp_1 = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
	IplImage* temp_2 = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 3);


	cvConvertScale(decoded_image, temp_1, 255.0/sl_params->proj_w, 0);
	colorizeWinter(temp_1, temp_2, mask);
	cvShowImageResampled("camWindow", temp_2, sl_params->window_w, sl_params->window_h);
	printf("Displaying the decoded columns; press any key (in 'camWindow') to continue.\n");
	cvWaitKey(0);


	// Free allocated resources.
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);

	// Return without errors.
	cvDestroyWindow("camWindow");
	return 0;
}

// Display a depth map.
int displayDepthMap(CvMat*& depth_map,
	IplImage*& mask,
struct slParams* sl_params){

	// Create a window to display depth map.
	cvNamedWindow("camWindow", CV_WINDOW_AUTOSIZE);

	cvWaitKey(1);

	// Allocate image arrays.
	IplImage* temp_1 = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
	IplImage* temp_2 = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 3);

	// Create depth map image (scaled to distance range).
	for(int r=0; r<sl_params->cam_h; r++){
		for(int c=0; c<sl_params->cam_w; c++){
			uchar* temp_1_data = (uchar*)(temp_1->imageData + r*temp_1->widthStep);
			uchar* mask_data   = (uchar*)(mask->imageData + r*mask->widthStep);
			if(mask_data[c])
				temp_1_data[c] = 
				255-int(255.0*(depth_map->data.fl[sl_params->cam_w*r+c]-sl_params->dist_range[0])/
				(sl_params->dist_range[1]-sl_params->dist_range[0]));
			else
				temp_1_data[c] = 0;
		}
	}

	// Display depth map.
	colorizeWinter(temp_1, temp_2, mask);
	cvShowImageResampled("camWindow", temp_2, sl_params->window_w, sl_params->window_h);
	printf("Displaying the depth map; press any key (in 'camWindow') to continue.\n");
	cvWaitKey(0);

	// Release allocated resources.
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_2);

	// Return without errors.
	cvDestroyWindow("camWindow");
	return 0;
}


// Display a depth map.
int displayEmphasizeDepthMap(CvMat*& depth_map,
struct slParams* sl_params, int num1, int num2){
	// 여기서 만큼은 마스크의 존재를 잊어버리자.
	// Create a window to display depth map.
	cvNamedWindow("depth_window", CV_WINDOW_AUTOSIZE);

	cvWaitKey(1);

	// Allocate image arrays.
	IplImage* temp_1 = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
	IplImage* temp_bg = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
	IplImage* temp_res = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 4);
	
	// 다른 색깔을 먹이자.
	cvSet(temp_bg, cvScalar(127));
	// Create depth map image (scaled to distance range).

	// '회색'을 기준으로 하자.
	for(int r=0; r<sl_params->cam_h; r++){
		for(int c=0; c<sl_params->cam_w; c++){
			uchar* temp_1_data = (uchar*)(temp_1->imageData + r*temp_1->widthStep);
			//uchar* mask_data   = (uchar*)(mask->imageData + r*mask->widthStep);
				temp_1_data[c] = 
				127-int(127.0*((depth_map->data.fl[sl_params->cam_w*r+c]-sl_params->dist_range[0])/
				(sl_params->dist_range[1]-sl_params->dist_range[0])));
		}
	}

	// Display depth map.
	//colorizeWinter(temp_1, temp_2, NULL);
	// 색깔이 달라지겠지
	cvMerge(temp_1, temp_bg, NULL, NULL, temp_res);

	cvShowImageResampled("depth_window", temp_1, sl_params->window_w, sl_params->window_h);

	printf("This is Showing depthmap for differential.");
	cvWaitKey(0);

	char str[200];
	sprintf(str, "%s\\%s\\%sdpmap_(%d-%d).png", sl_params->outdir, sl_params->object, sl_params->object, num1,num2);
	cvSaveImage(str,temp_1);

	// Release allocated resources.
	cvReleaseImage(&temp_1);
	cvReleaseImage(&temp_bg);
	cvReleaseImage(&temp_res);

	// Return without errors.
	cvDestroyWindow("depth_window");
	return 0;
}




// Reconstruct the point cloud and the depth map from a structured light sequence.
int reconstructStructuredLight(struct slParams* sl_params, 
struct slCalib* sl_calib,
	IplImage*& texture_image,
	IplImage*& gray_decoded_image,
	IplImage*& gray_mask,
	CvMat*&    points,
	CvMat*&    colors,
	CvMat*&    depth_map,
	CvMat*&    mask,
	int Angle){

		int p_num;
		switch(Angle){
		case 0		:	p_num=0; break;
		case 26		:	p_num=1; break;
		case 45		:	p_num=2; break;
		case 63		:	p_num=3; break;
		case 90		:	p_num=4; break;
		case 116	:	p_num=5; break;
		case 135	:	p_num=6; break;
		case 154	:	p_num=7; break;
		default		:	break;
		}


		// Define pointers to various image data elements (for fast pixel access).
		int cam_nelems                 = sl_params->cam_w*sl_params->cam_h;
		int proj_nelems                = sl_params->proj_w*sl_params->proj_h;
		uchar*  background_mask_data   = (uchar*)sl_calib->background_mask->imageData;
		int     background_mask_step   = sl_calib->background_mask->widthStep/sizeof(uchar);
		uchar*  gray_mask_data         = (uchar*)gray_mask->imageData;
		int     gray_mask_step         = gray_mask->widthStep/sizeof(uchar);
		ushort* gray_decoded_image_data = (ushort*)gray_decoded_image->imageData;
		int     gray_decoded_image_step = gray_decoded_image->widthStep/sizeof(ushort);

		// Create a temporary copy of the background depth map.
		CvMat* background_depth_map = cvCloneMat(sl_calib->background_depth_map);

		// By default, disable all pixels.
		cvZero(mask);

		// Reconstruct point cloud and depth map.
		for(int r=0; r<sl_params->cam_h-0; r++){
			for(int c=0; c<sl_params->cam_w-0; c++){
				// Reconstruct current point, if mask is non-zero.
				if(gray_mask_data[r*gray_mask_step+c]){
					float point[3];
					float depth;
					float q[3], v[3], w[4];
					int rc = (sl_params->cam_w)*r+c;
					for(int i=0; i<3; i++){
						q[i] = sl_calib->cam_center->data.fl[i];
						v[i] = sl_calib->cam_rays->data.fl[rc+cam_nelems*i];
					}
					int corresponse = gray_decoded_image_data[r*gray_decoded_image_step+c];
					for(int i=0; i<4; i++)
						w[i] = sl_calib->proj_planes[p_num]->data.fl[4*corresponse+i];
					intersectLineWithPlane3D(q, v, w, point, depth);
					depth_map->data.fl[sl_params->cam_w*r+c] = depth;
					for(int i=0; i<3; i++)
						points->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = point[i];

					// Assign color using provided texture image.
					// Note: Color channels are ordered as RGB, rather than OpenCV's default BGR.
					uchar* texture_image_data = (uchar*)(texture_image->imageData + r*texture_image->widthStep);
					for(int i=0; i<3; i++)
						colors->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = (float)texture_image_data[3*c+(2-i)]/(float)255.0;
					mask->data.fl[sl_params->cam_w*r+c] = 1;

					// Reject any points outside near/far clipping planes.
					if(depth_map->data.fl[sl_params->cam_w*r+c] < sl_params->dist_range[0] ||
						depth_map->data.fl[sl_params->cam_w*r+c] > sl_params->dist_range[1]){
							gray_mask_data[r*gray_mask_step+c] = 0;
							mask->data.fl[sl_params->cam_w*r+c] = 0;
							depth_map->data.fl[sl_params->cam_w*r+c] = 0;
							for(int i=0; i<3; i++)
								points->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = 0;
							for(int i=0; i<3; i++)
								colors->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = 0;
					}

					// Reject background points.
					// Note: Currently only uses depth to determine foreground vs. background pixels.
					float depth_difference = 
						background_depth_map->data.fl[sl_params->cam_w*r+c] - 
						depth_map->data.fl[sl_params->cam_w*r+c];
					if(depth_difference < sl_params->background_depth_thresh && 
						gray_mask_data[r*gray_mask_step+c] && 
						background_mask_data[r*background_mask_step+c]){
							gray_mask_data[r*gray_mask_step+c] = 0;
							mask->data.fl[sl_params->cam_w*r+c] = 0;
							depth_map->data.fl[sl_params->cam_w*r+c] = 0;
							for(int i=0; i<3; i++)
								points->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = 0;
							for(int i=0; i<3; i++)
								colors->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = 0;
					}
				}
			}
		}

		// Release allocated resources.
		cvReleaseMat(&background_depth_map);

		// Return without errors.
		return 0;
}





// Run the background capture (used to eliminate background points from reconstructions).
int runBackgroundCapture(CvCapture* capture, 
struct slParams* sl_params, 
struct slCalib* sl_calib,int Angle){

	// Generate Gray codes.
	IplImage** proj_gray_codes = NULL;
	int r_plane, n_plane, n_shift;

	generateGrayCodes(sl_params->proj_w, sl_params->proj_h, proj_gray_codes, 
		r_plane,n_plane,n_shift,Angle);

	// Capture live image stream (e.g., for adjusting object placement).
	printf("Remove object, then press any key (in 'camWindow') to scan.\n");
	camPreview(capture, sl_params, sl_calib);

	// Illuminate the background using the Gray code sequence.
	printf("Displaying the structured light sequence...\n");
	IplImage** cam_gray_codes = NULL;
	slScan(capture, proj_gray_codes, cam_gray_codes, n_plane,sl_params, sl_calib);

	// Save white image for background model.
	cvCopyImage(cam_gray_codes[0], sl_calib->background_image);

	// Decode the structured light sequence.
	printf("Decoding the structured light sequence...\n");
	IplImage* gray_decoded_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_16U, 1);
	decodeGrayCodes(sl_params->proj_w, sl_params->proj_h,
		cam_gray_codes, gray_decoded_image, sl_calib->background_mask,
		n_plane,r_plane,n_shift,sl_params->thresh);


	// Reconstruct the point cloud and depth map.
	printf("Reconstructing the point cloud and the depth map...\n");
	CvMat *points  = cvCreateMat(3, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
	CvMat *colors  = cvCreateMat(3, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
	CvMat *mask    = cvCreateMat(1, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
	reconstructStructuredLight(sl_params,sl_calib,cam_gray_codes[0],gray_decoded_image,sl_calib->background_mask, 
		points, colors, sl_calib->background_depth_map,mask,Angle);

	// Free allocated resources.
	cvReleaseImage(&gray_decoded_image);
	cvReleaseMat(&points);
	cvReleaseMat(&colors);
	cvReleaseMat(&mask);
	for(int i=0; i<(n_plane+1); i++)
		cvReleaseImage(&proj_gray_codes[i]);
	delete[] proj_gray_codes;
	for(int i=0; i<2*(n_plane+1); i++)
		cvReleaseImage(&cam_gray_codes[i]);
	delete[] cam_gray_codes;

	// Return without errors.
	return 0;
}


// Run the structured light scanner.
int All_runStructuredLight(CvCapture* capture, 
struct slParams* sl_params, 
struct slCalib* sl_calib, 
	int scan_index){

		// Create output directory (if output enabled).
		char str[1024], outputDir[1024];
		if(sl_params->save){
			sprintf(outputDir, "%s\\%s\\%0.2d", sl_params->outdir, sl_params->object, scan_index);
			_mkdir(outputDir);
		}
		// Save the current calibration parameters (if output enabled).
		if(sl_calib->cam_intrinsic_calib && sl_params->save){
			sprintf(str,"%s\\cam_intrinsic.xml", outputDir);	
			cvSave(str, sl_calib->cam_intrinsic);
			sprintf(str,"%s\\cam_distortion.xml", outputDir);
			cvSave(str, sl_calib->cam_distortion);
		}
		if(sl_calib->proj_intrinsic_calib && sl_params->save){
			sprintf(str,"%s\\proj_intrinsic.xml", outputDir);	
			cvSave(str, sl_calib->proj_intrinsic);
			sprintf(str,"%s\\proj_distortion.xml", outputDir);
			cvSave(str, sl_calib->proj_distortion);
		}
		if(sl_calib->procam_extrinsic_calib && sl_params->save){
			sprintf(str,"%s\\cam_extrinsic.xml", outputDir);	
			cvSave(str, sl_calib->cam_extrinsic);
			sprintf(str,"%s\\proj_extrinsic.xml", outputDir);	
			cvSave(str, sl_calib->proj_extrinsic);
		}
		if(sl_params->save){
			sprintf(str,"%s\\config.xml", outputDir);
			writeConfiguration(str, sl_params);
		}


		int Angle;
		int gray_n_shift,r_plane,n_plane;
		IplImage* gray_decoded_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_16U, 1);
		IplImage* gray_mask[9];
		IplImage *depth_img[9];
		CvMat *points[9];
		CvMat *colors[9];
		CvMat *depth_map[9];
		CvMat *mask[9];
		for(int j=0;j<9;j++){
			depth_img[j]	=	cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_32F,  1);
			gray_mask[j]	=	cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U,  1);
			points[j]		=	cvCreateMat(3, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
			mask[j]			=	cvCreateMat(1, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
			depth_map[j]	=	cvCreateMat(sl_params->cam_h, sl_params->cam_w, CV_32FC1);
			colors[j]		=	cvCreateMat(3, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
			cvSet(depth_img[j], cvScalar(0));
		}
		IplImage** proj_gray_codes = NULL;
		IplImage** cam_gray_codes = NULL;
		// 8방향 스캐닝 시작!
		camPreview(capture, sl_params, sl_calib);
		for(int j=0;j<8;j++){
			switch(j){
			case 0	:	Angle=0;	break;
			case 1	:	Angle=26;	break;
			case 2	:	Angle=45;	break;
			case 3	:	Angle=63;	break;
			case 4	:	Angle=90;	break;
			case 5	:	Angle=116;	break;
			case 6	:	Angle=135;	break;
			case 7	:	Angle=154;	break;
			default	:				break;
			}
			// Generate Gray codes.
			generateGrayCodes(sl_params->proj_w, sl_params->proj_h, proj_gray_codes,
				r_plane,n_plane,gray_n_shift,Angle);

			// Capture live image stream (e.g., for adjusting object placement).
			printf("Position object, then press any key (in 'camWindow') to scan.\n");
			
			// Illuminate the object using the Gray code sequence.
			printf("Displaying the structured light sequence...\n");
			slScan(capture, proj_gray_codes, cam_gray_codes, n_plane,sl_params, sl_calib);

			// Save the capture image sequence (if enabled).
			if(sl_params->save){
				printf("Saving the structured light sequence...\n");
				for(int i=0; i<2*(n_plane+1); i++){
					sprintf(str, "%s\\%0.2d\\%d.png", outputDir, i,j);
					cvSaveImage(str, cam_gray_codes[i]);
				}
			}
			// Decode the structured light sequence.
			printf("Decoding the structured light sequence...\n");
			decodeGrayCodes(sl_params->proj_w, sl_params->proj_h,
				cam_gray_codes, gray_decoded_image, gray_mask[j],
				n_plane,r_plane,gray_n_shift,sl_params->thresh);

			uchar*  gray_mask_data         = (uchar*)gray_mask[j]->imageData;
			int     gray_mask_step         = gray_mask[j]->widthStep/sizeof(uchar);


			printf("Reconstructing the point cloud and the depth map...\n");
			reconstructStructuredLight(sl_params, sl_calib, cam_gray_codes[0],
				gray_decoded_image, gray_mask[j],
				points[j], colors[j], depth_map[j],mask[j],Angle);

			// 각 포인트를 일단 임시저장. 추후 딜릿.
			printf("Saving the point cloud...\n");
			sprintf(str, "%s\\%s\\%s_(%d).wrl", sl_params->outdir, sl_params->object, sl_params->object, Angle);
			if(savePointsVRML(str, points[j], NULL, colors[j], mask[j])){
				printf("Scanning was not successful and must be repeated!\n");
				return -1;
			}

			for(int i=0; i<(n_plane+1); i++)
				cvReleaseImage(&proj_gray_codes[i]);
			delete[] proj_gray_codes;
			for(int i=0; i<2*(n_plane+1); i++)
				cvReleaseImage(&cam_gray_codes[i]);
			delete[] cam_gray_codes;
		}

		IplImage* edge_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
		IplImage* binary_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
		IplImage * cam_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 4);

		// Depth Map을 확인하기 위해 임시로 삽입한 코드다.
		/*for(int i = 0; i<8; i++){
			displayDepthMap(depth_map[i], gray_mask[i], sl_params);
		}*/

		Canny(capture,cam_image, sl_params, edge_image, binary_image);
		//Canny(capture, cam_image, sl_params, edge_image);

		cvSet(gray_mask[8], cvScalar(0));
		cvZero(mask[8]);	cvZero(depth_map[8]);	cvZero(points[8]);	cvZero(colors[8]);
		
		int cam_nelems		= sl_params->cam_w*sl_params->cam_h;
		int mask_step		= edge_image->widthStep/sizeof(uchar);
		uchar* edge_data	= (uchar*)edge_image->imageData;
		int edge_step		= edge_image->widthStep/sizeof(uchar);
		IplImage* edge_plane	= cvCreateImage(cvSize(sl_params->cam_w-2, sl_params->cam_h-2), IPL_DEPTH_8U, 4);	 // 최종 Depth 합산 이미지.
		IplImage* R_plane		= cvCreateImage(cvSize(sl_params->cam_w-2, sl_params->cam_h-2), IPL_DEPTH_8U, 1);
		IplImage* B_plane		= cvCreateImage(cvSize(sl_params->cam_w-2, sl_params->cam_h-2), IPL_DEPTH_8U, 1);
		IplImage* G_plane		= cvCreateImage(cvSize(sl_params->cam_w-2, sl_params->cam_h-2), IPL_DEPTH_8U, 1);
		cvSet(R_plane, cvScalar(0));	cvSet(G_plane, cvScalar(0));	cvSet(B_plane, cvScalar(0));
	
		uchar* R_data	= (uchar*)R_plane->imageData;
		uchar* G_data	= (uchar*)G_plane->imageData;
		uchar* B_data	= (uchar*)B_plane->imageData;
		int step		= B_plane->widthStep/sizeof(uchar);
		int select;

		/// edge와 Sobel Mask 에 의한 취사선택.
		// 그리고 각 Angle에 대해 삭깔을 부여한다!
		// 고로 이것으로 depth_map[8]을 완성.
		for(int r=2;r<sl_params->cam_h-2;r++){
			for(int c=2;c<sl_params->cam_w-2;c++){
				switch(edge_data[r*edge_step+c]){
				case 0		:	{select=4;	R_data[(r-2)*step+(c-2)]=255;	G_data[(r-2)*step+(c-2)]=0;		B_data[(r-2)*step+(c-2)]=0;}	break;
				case 26		:	{select=5;	R_data[(r-2)*step+(c-2)]=0;		G_data[(r-2)*step+(c-2)]=255;	B_data[(r-2)*step+(c-2)]=0;}	break;
				case 45		:	{select=6;	R_data[(r-2)*step+(c-2)]=0;		G_data[(r-2)*step+(c-2)]=0;		B_data[(r-2)*step+(c-2)]=255;}	break;
				case 63		:	{select=7;	R_data[(r-2)*step+(c-2)]=255;	G_data[(r-2)*step+(c-2)]=228;	B_data[(r-2)*step+(c-2)]=0;}	break;
				case 90		:	{select=0;	R_data[(r-2)*step+(c-2)]=255;	G_data[(r-2)*step+(c-2)]=94;	B_data[(r-2)*step+(c-2)]=0;}	break;
				case 116	:	{select=1;	R_data[(r-2)*step+(c-2)]=0;		G_data[(r-2)*step+(c-2)]=216;	B_data[(r-2)*step+(c-2)]=255;}	break;
				case 135	:	{select=2;	R_data[(r-2)*step+(c-2)]=95;	G_data[(r-2)*step+(c-2)]=0;		B_data[(r-2)*step+(c-2)]=255;}	break;
				case 154	:	{select=3;	R_data[(r-2)*step+(c-2)]=200;	G_data[(r-2)*step+(c-2)]=100;	B_data[(r-2)*step+(c-2)]=255;}	break;
				default		:	break;
				}
				uchar*  gray_mask_data						= (uchar*)gray_mask[select]->imageData;
				uchar*  T_gray_mask_data					= (uchar*)gray_mask[8]->imageData;
				float*  depth_data							= (float*)depth_img[select]->imageData;
				int gray_mask_step							= gray_mask[select]->widthStep/sizeof(uchar);
				int T_gray_mask_step						= gray_mask[8]->widthStep/sizeof(uchar);
				int depth_step								= depth_img[select]->widthStep/sizeof(float);
				T_gray_mask_data[r*T_gray_mask_step+c]		= gray_mask_data[r*gray_mask_step+c];
				mask[8]->data.fl[sl_params->cam_w*r+c]		= mask[select]->data.fl[sl_params->cam_w*r+c];
				depth_map[8]->data.fl[sl_params->cam_w*r+c] = depth_map[select]->data.fl[sl_params->cam_w*r+c];
				depth_data[r*depth_step+c]					= depth_map[select]->data.fl[sl_params->cam_w*r+c];
				// Count num of the Scanned Pixels
				for(int i=0; i<3; i++){
					points[8]->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = points[select]->data.fl[sl_params->cam_w*r+c+cam_nelems*i];
					colors[8]->data.fl[sl_params->cam_w*r+c+cam_nelems*i] = colors[select]->data.fl[sl_params->cam_w*r+c+cam_nelems*i];
				}
			}
		}
		cvMerge(B_plane, G_plane, R_plane, NULL,edge_plane);
		cvShowImage("8 colored difftent Angle", edge_plane);
		cvWaitKey(1);

		IplImage* B_depth		= cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
		IplImage* G_depth		= cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
		IplImage* R_depth		= cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
		IplImage* D_edge		= cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
		IplImage* color_depth	= cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 4);
		cvSet(R_depth, cvScalar(0));	cvSet(G_depth, cvScalar(0));	cvSet(B_depth, cvScalar(0));
	
		uchar* BD_data	= (uchar*)B_depth->imageData;
		uchar* GD_data	= (uchar*)G_depth->imageData;
		uchar* RD_data	= (uchar*)R_depth->imageData;
		int color_step		= B_depth->widthStep/sizeof(uchar);
		uchar* binary_data	= (uchar*)binary_image->imageData;
		int binary_step		= binary_image->widthStep/sizeof(uchar);


		int num1,num2;
		for(int num1=0;num1<8;num1++)
			for(int num2=0;num2<8;num2++){
				depthMapCompare(depth_map[num1], depth_map[num2], gray_mask[0], sl_params, num1, num2);

			cvSet(G_depth, cvScalar(0));
			cvSet(B_depth, cvScalar(0));
			cvSet(R_depth, cvScalar(0));
			cvSet(D_edge, cvScalar(255));

			int		depth_step	= depth_img[0]->widthStep/sizeof(float);
			int		depth_step2	= D_edge->widthStep/sizeof(uchar);
			float*  depth_data1	= (float*)depth_img[num1]->imageData;
			float*  depth_data2	= (float*)depth_img[num2]->imageData;
			uchar*  depth_data3	= (uchar*)D_edge->imageData;
			switch(num1){
				case 0		:	select=0;	break;
				case 1		:	select=26;	break;
				case 2		:	select=45;	break;
				case 3		:	select=63;	break;
				case 4		:	select=90;	break;
				case 5		:	select=116;	break;
				case 6		:	select=135;	break;
				case 7		:	select=154;	break;
				default		:	break;
				}
			for(int r=2;r<sl_params->cam_h-2;r++){
				for(int c=2;c<sl_params->cam_w-2;c++){
					if((edge_data[r*edge_step+c]==select)&&(binary_data[r*edge_step+c]==255)){
						depth_data3[r*depth_step2+c] = 0;
						if((depth_data1[r*depth_step+c]!=0)&&(depth_data2[r*depth_step+c]!=0))
							RD_data[r*color_step+c] = (depth_data1[r*depth_step+c]-depth_data2[r*depth_step+c])+200;
						if((depth_data1[r*depth_step+c]==0)&&(depth_data2[r*depth_step+c]!=0))
							BD_data[r*color_step+c] =255;
						if((depth_data1[r*depth_step+c]!=0)&&(depth_data2[r*depth_step+c]==0))
							GD_data[r*color_step+c] =255;
						if((depth_data1[r*depth_step+c]==0)&&(depth_data2[r*depth_step+c]==0)){
							BD_data[r*color_step+c] =255;GD_data[r*color_step+c] =255;
						}
					}
				}
			}
			cvMerge(B_depth, G_depth, R_depth, NULL,color_depth);
			printf("Saving the diffrent depth map...\n");
			sprintf(str, "%s\\%s\\%s_%0.2d(%d-%d).png", sl_params->outdir, sl_params->object, sl_params->object, scan_index,num1,num2);
			cvSaveImage(str,color_depth);
			cvShowImage("depth_map", color_depth);
			cvWaitKey(1);
			sprintf(str, "%s\\depth_map(%d).png", outputDir,select);
			cvSaveImage(str,D_edge);
		}
		cvDestroyWindow("8 colored difftent Angle");
		cvDestroyWindow("depth_map");

		if(sl_params->display)
			displayDepthMap(depth_map[8], gray_mask[8], sl_params);
		if(sl_params->save){
			printf("\nSaving the depth map...\n");
			IplImage* depth_map_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
			for(int r=2; r<sl_params->cam_h-2; r++){
				for(int c=2; c<sl_params->cam_w-2; c++){
					char* depth_map_image_data = (char*)(depth_map_image->imageData + r*depth_map_image->widthStep);
					if(mask[8]->data.fl[sl_params->cam_w*r+c])
						depth_map_image_data[c] = 
						255-int(255*(depth_map[8]->data.fl[sl_params->cam_w*r+c]-sl_params->dist_range[0])/
						(sl_params->dist_range[1]-sl_params->dist_range[0]));
					else
						depth_map_image_data[c] = 0;
				}
			}
			CvMat* dist_range = cvCreateMat(1, 2, CV_32FC1);
			cvmSet(dist_range, 0, 0, sl_params->dist_range[0]);
			cvmSet(dist_range, 0, 1, sl_params->dist_range[1]);
			char str[1024];
			sprintf(str, "%s\\depth_map.png", outputDir);
			cvSaveImage(str, depth_map_image);
			sprintf(str, "%s\\depth_map_range.xml", outputDir);
			cvSave(str, dist_range);
			cvReleaseImage(&depth_map_image);
			cvReleaseMat(&dist_range);
		}


		// Save the point cloud.
		printf("Saving the point cloud...\n");
		sprintf(str, "%s\\%s\\%s_%0.2d.wrl", sl_params->outdir, sl_params->object, sl_params->object, scan_index);
		if(savePointsVRML(str, points[8], NULL, colors[8], mask[8])){
			printf("Scanning was not successful and must be repeated!\n");
			return -1;
		}

		// Free allocated resources.
		cvReleaseImage(&edge_image);
		cvReleaseImage(&cam_image);
		cvReleaseImage(&gray_decoded_image);
		cvReleaseImage(&R_plane);
		cvReleaseImage(&G_plane);
		cvReleaseImage(&B_plane);
		cvReleaseImage(&edge_plane);
		for(int i=0;i<9;i++){
			cvReleaseImage(&gray_mask[i]);
			cvReleaseMat(&points[i]);
			cvReleaseMat(&colors[i]);
			cvReleaseMat(&depth_map[i]);
			cvReleaseMat(&mask[i]);
		}
		// Return without errors.
		return 0;
}


// 비교연산 알고리즘.
int depthMapCompare(CvMat * src1, CvMat* src2, IplImage * gray_mask, struct slParams * sl_params, int num1, int num2){
	//CvMat * temp1 = cvCreateMat(sl_params->cam_h, sl_params->cam_w, CV_32FC1);
	CvMat * temp2 = cvCreateMat(sl_params->cam_h, sl_params->cam_w, CV_32FC1);
	//cvSet(temp1, cvScalar(250));
	cvSet(temp2, cvScalar(0));
	cvSub(src1, src2, temp2);
	//cvAdd(temp2, temp1, temp1);
	

	displayEmphasizeDepthMap(temp2, sl_params, num1, num2);

	//cvReleaseMat(&temp1);
	cvReleaseMat(&temp2);

	return 0;
}


// Run the structured light scanner.
int runStructuredLight(CvCapture* capture, 
struct slParams* sl_params, 
struct slCalib* sl_calib, 
	int scan_index,int Angle){

		// Generate Gray codes.
		IplImage** proj_gray_codes = NULL;
		int gray_n_shift,r_plane,n_plane;

		generateGrayCodes(sl_params->proj_w, sl_params->proj_h, proj_gray_codes,
			r_plane,n_plane,gray_n_shift,Angle);

		// Capture live image stream (e.g., for adjusting object placement).
		printf("Position object, then press any key (in 'camWindow') to scan.\n");
		camPreview(capture, sl_params, sl_calib);

		// Illuminate the object using the Gray code sequence.
		printf("Displaying the structured light sequence...\n");
		IplImage** cam_gray_codes = NULL;

		slScan(capture, proj_gray_codes, cam_gray_codes, n_plane,sl_params, sl_calib);

		// Create output directory (if output enabled).
		char str[1024], outputDir[1024];
		if(sl_params->save){
			sprintf(outputDir, "%s\\%s\\%0.2d", sl_params->outdir, sl_params->object, scan_index);
			_mkdir(outputDir);
		}

		// Save the current calibration parameters (if output enabled).
		if(sl_calib->cam_intrinsic_calib && sl_params->save){
			sprintf(str,"%s\\cam_intrinsic.xml", outputDir);	
			cvSave(str, sl_calib->cam_intrinsic);
			sprintf(str,"%s\\cam_distortion.xml", outputDir);
			cvSave(str, sl_calib->cam_distortion);
		}
		if(sl_calib->proj_intrinsic_calib && sl_params->save){
			sprintf(str,"%s\\proj_intrinsic.xml", outputDir);	
			cvSave(str, sl_calib->proj_intrinsic);
			sprintf(str,"%s\\proj_distortion.xml", outputDir);
			cvSave(str, sl_calib->proj_distortion);
		}
		if(sl_calib->procam_extrinsic_calib && sl_params->save){
			sprintf(str,"%s\\cam_extrinsic.xml", outputDir);	
			cvSave(str, sl_calib->cam_extrinsic);
			sprintf(str,"%s\\proj_extrinsic.xml", outputDir);	
			cvSave(str, sl_calib->proj_extrinsic);
		}
		if(sl_params->save){
			sprintf(str,"%s\\config.xml", outputDir);
			writeConfiguration(str, sl_params);
		}

		// Save the capture image sequence (if enabled).
		if(sl_params->save){
			printf("Saving the structured light sequence...\n");
			for(int i=0; i<2*(n_plane+1); i++){
				sprintf(str, "%s\\%0.2d.png", outputDir, i);
				cvSaveImage(str, cam_gray_codes[i]);
			}
		}

		// Decode the structured light sequence.
		printf("Decoding the structured light sequence...\n");
		IplImage* gray_decoded_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_16U, 1);
		IplImage* gray_mask         = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U,  1);
		decodeGrayCodes(sl_params->proj_w, sl_params->proj_h,
			cam_gray_codes, gray_decoded_image, sl_calib->background_mask,
			n_plane,r_plane,gray_n_shift,sl_params->thresh);

		// Display and save the correspondences.
		if(sl_params->display)
			displayDecodingResults(gray_decoded_image, gray_mask, sl_params);

		// Reconstruct the point cloud and depth map.
		printf("Reconstructing the point cloud and the depth map...\n");
		CvMat *points    = cvCreateMat(3, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
		CvMat *colors    = cvCreateMat(3, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
		CvMat *depth_map = cvCreateMat(sl_params->cam_h, sl_params->cam_w, CV_32FC1);
		CvMat *mask      = cvCreateMat(1, sl_params->cam_h*sl_params->cam_w, CV_32FC1);
		reconstructStructuredLight(sl_params, sl_calib, cam_gray_codes[0],
			gray_decoded_image, sl_calib->background_mask,
			points, colors, sl_calib->background_depth_map,mask,Angle);

		// Display and save the depth map.
		if(sl_params->display)
			displayDepthMap(depth_map, gray_mask, sl_params);
		if(sl_params->save){
			printf("Saving the depth map...\n");
			IplImage* depth_map_image = cvCreateImage(cvSize(sl_params->cam_w, sl_params->cam_h), IPL_DEPTH_8U, 1);
			for(int r=0; r<sl_params->cam_h; r++){
				for(int c=0; c<sl_params->cam_w; c++){
					char* depth_map_image_data = (char*)(depth_map_image->imageData + r*depth_map_image->widthStep);
					if(mask->data.fl[sl_params->cam_w*r+c])
						depth_map_image_data[c] = 
						255-int(255*(depth_map->data.fl[sl_params->cam_w*r+c]-sl_params->dist_range[0])/
						(sl_params->dist_range[1]-sl_params->dist_range[0]));
					else
						depth_map_image_data[c] = 0;
				}
			}
			CvMat* dist_range = cvCreateMat(1, 2, CV_32FC1);
			cvmSet(dist_range, 0, 0, sl_params->dist_range[0]);
			cvmSet(dist_range, 0, 1, sl_params->dist_range[1]);
			char str[1024];
			sprintf(str, "%s\\depth_map.png", outputDir);
			cvSaveImage(str, depth_map_image);
			sprintf(str, "%s\\depth_map_range.xml", outputDir);
			cvSave(str, dist_range);
			cvReleaseImage(&depth_map_image);
			cvReleaseMat(&dist_range);
		}

		// Save the texture map.
		printf("Saving the texture map...\n");
		sprintf(str, "%s\\%s\\%s_%0.2d.png", sl_params->outdir, sl_params->object, sl_params->object, scan_index);
		cvSaveImage(str, cam_gray_codes[0]);

		// Save the point cloud.
		printf("Saving the point cloud...\n");
		sprintf(str, "%s\\%s\\%s_%0.2d.wrl", sl_params->outdir, sl_params->object, sl_params->object, scan_index);
		if(savePointsVRML(str, points, NULL, colors, mask)){
			printf("Scanning was not successful and must be repeated!\n");
			return -1;
		}

		// Free allocated resources.
		cvReleaseImage(&gray_decoded_image);
		cvReleaseImage(&gray_mask);
		cvReleaseMat(&points);
		cvReleaseMat(&colors);
		cvReleaseMat(&depth_map);
		cvReleaseMat(&mask);
		for(int i=0; i<(n_plane+1); i++)
			cvReleaseImage(&proj_gray_codes[i]);
		delete[] proj_gray_codes;
		for(int i=0; i<2*(n_plane+1); i++)
			cvReleaseImage(&cam_gray_codes[i]);
		delete[] cam_gray_codes;

		// Return without errors.
		return 0;
}


int GaussianBlur(IplImage*& cam_gray,int width,int height){

	int step = cam_gray->widthStep/sizeof(uchar);
	int gaussianMask[5][5];
	int newPixel,rowOffset,colOffset,rowTotal,colTotal,iOffset;
	uchar* data = (uchar*)cam_gray->imageData;
	/* Declare Gaussian mask */
	gaussianMask[0][0] = 2;		gaussianMask[0][1] = 4;		gaussianMask[0][2] = 5;		gaussianMask[0][3] = 4;		gaussianMask[0][4] = 2;
	gaussianMask[1][0] = 4;		gaussianMask[1][1] = 9;		gaussianMask[1][2] = 12;	gaussianMask[1][3] = 9;		gaussianMask[1][4] = 4;	
	gaussianMask[2][0] = 5;		gaussianMask[2][1] = 12;	gaussianMask[2][2] = 15;	gaussianMask[2][3] = 12;	gaussianMask[2][4] = 2;	
	gaussianMask[3][0] = 4;		gaussianMask[3][1] = 9;		gaussianMask[3][2] = 12;	gaussianMask[3][3] = 9;		gaussianMask[3][4] = 4;	
	gaussianMask[4][0] = 2;		gaussianMask[4][1] = 4;		gaussianMask[4][2] = 5;		gaussianMask[4][3] = 4;		gaussianMask[4][4] = 2;	


	/* Gaussian Blur */
	for (int row = 2; row < height-2; row++) {
		for (int col = 2; col < width-2; col++) {
			newPixel = 0;
			for (rowOffset=-2; rowOffset<=2; rowOffset++) {
				for (colOffset=-2; colOffset<=2; colOffset++) {
					rowTotal = row + rowOffset;
					colTotal = col + colOffset;
					iOffset = rowTotal*step + colTotal;
					newPixel += data[iOffset] * gaussianMask[2 + rowOffset][2 + colOffset];
				}
			}
			int i = row*step + col;
			data[i] = newPixel / 159;
		}
	}
	return 0;
}

int C_SobelMask(IplImage* cam_input,IplImage*& edge_image,IplImage*& binary_image,int width,int height){

	int step = cam_input->widthStep/sizeof(uchar);
	uchar* data = (uchar*)cam_input->imageData;
	int edge_step = edge_image->widthStep/sizeof(uchar);
	uchar* edge_data = (uchar*)edge_image->imageData;
	int binary_step = binary_image->widthStep/sizeof(uchar);
	uchar* binary_data = (uchar*)binary_image->imageData;

	int GxMask[5][5],GyMask[5][5];
	double Gx,Gy, thisAngle;
	int newAngle,rowOffset,colOffset,rowTotal,colTotal,iOffset,i;

	/* Declare Sobel masks */
	GxMask[0][0] = 1;		GxMask[0][1] = 2;		GxMask[0][2] = 0;		GxMask[0][3] = -2;		GxMask[0][4] = -1;
	GxMask[1][0] = 4;		GxMask[1][1] = 8;		GxMask[1][2] = 0;		GxMask[1][3] = -8;		GxMask[1][4] = -4;	
	GxMask[2][0] = 6;		GxMask[2][1] = 12;		GxMask[2][2] = 0;		GxMask[2][3] = -12;		GxMask[2][4] = -6;	
	GxMask[3][0] = 4;		GxMask[3][1] = 8;		GxMask[3][2] = 0;		GxMask[3][3] = -8;		GxMask[3][4] = -4;	
	GxMask[4][0] = 1;		GxMask[4][1] = 2;		GxMask[4][2] = 0;		GxMask[4][3] = -2;		GxMask[4][4] = -1;	

	GyMask[0][0] = -1;		GyMask[0][1] = -4;		GyMask[0][2] = -6;		GyMask[0][3] = -4;		GyMask[0][4] = -1;
	GyMask[1][0] = -2;		GyMask[1][1] = -8;		GyMask[1][2] = -12;		GyMask[1][3] = -8;		GyMask[1][4] = -2;	
	GyMask[2][0] = 0;		GyMask[2][1] = 0;		GyMask[2][2] = 0;		GyMask[2][3] = 0;		GyMask[2][4] = 0;	
	GyMask[3][0] = 2;		GyMask[3][1] = 8;		GyMask[3][2] = 12;		GyMask[3][3] = 8;		GyMask[3][4] = 2;	
	GyMask[4][0] = 1;		GyMask[4][1] = 4;		GyMask[4][2] = 6;		GyMask[4][3] = 4;		GyMask[4][4] = 1;	


	for (int row = 2; row < height-2; row++) {
		for (int col = 2; col < width-2; col++) {
			i = row*step + col;
			Gx = 0;	Gy = 0;
			for (rowOffset=-2; rowOffset<=2; rowOffset++) {
				for (colOffset=-2; colOffset<=2; colOffset++) {
					rowTotal = row + rowOffset;
					colTotal = col + colOffset;
					iOffset = rowTotal*width + colTotal;
					Gx = Gx + data[iOffset] * GxMask[rowOffset + 2][colOffset + 2];
					Gy = Gy + data[iOffset] * GyMask[rowOffset + 2][colOffset + 2];
				}
			}
			thisAngle = (atan2(Gx,Gy)/3.14159) * 180.0;
			if ( ( (thisAngle < 13.28) && (thisAngle > -13.28  ) ) || (thisAngle > 166.72) || (thisAngle < -166.72) )
				newAngle = 0;
			if ( ( (thisAngle > 13.28) && (thisAngle < 35.78) ) || ( (thisAngle < -144.22) && (thisAngle > -166.72) ) )
				newAngle = 26;
			if ( ( (thisAngle > 35.78) && (thisAngle < 54.22) ) || ( (thisAngle < -125.78) && (thisAngle > -144.22) ) )
				newAngle = 45;
			if ( ( (thisAngle > 54.22) && (thisAngle < 76.72) ) || ( (thisAngle < -103.28) && (thisAngle > -125.78) ) )
				newAngle = 63;
			if ( ( (thisAngle > 76.72) && (thisAngle < 103.28) ) || ( (thisAngle < -76.72) && (thisAngle > -103.28) ) )
				newAngle = 90;
			if ( ( (thisAngle > 103.28) && (thisAngle < 125.78) ) || ( (thisAngle < -54.22) && (thisAngle > -76.72) ) )
				newAngle = 116;
			if ( ( (thisAngle > 125.78) && (thisAngle < 144.22) ) || ( (thisAngle < -35.78) && (thisAngle > -54.22) ) )
				newAngle = 135;
			if ( ( (thisAngle > 144.22) && (thisAngle < 166.72) ) || ( (thisAngle < -13.28) && (thisAngle > -35.78) ) )
				newAngle = 154;
			if(25*25<sqrt(pow(Gx,2)+pow(Gy,2))){
				edge_data[row*edge_step+col] = newAngle;
				binary_data[row*binary_step+col] = 255;
			}
			else
				edge_data[row*edge_step+col] = 0;
		}
	}
	return 0;
}

int Canny(CvCapture* capture, IplImage* cam_image, struct slParams* sl_params, IplImage* edge_image,IplImage* binary_image){

	IplImage* proj_frame = cvCreateImage(cvSize(sl_params->proj_w, sl_params->proj_h), IPL_DEPTH_8U, 1);
	cvSet(proj_frame, cvScalar(255));
	cvScale(proj_frame, proj_frame, 100./sl_params->proj_gain);
	cvShowImage("projWindow",proj_frame);

	IplImage* cam_frame  = cvQueryFrame(capture);
	cvNamedWindow("camWindow", CV_WINDOW_AUTOSIZE);
	cvWaitKey(1);
	// 이미지 복사.
	cam_image = cvCloneImage(cam_frame);
	IplImage* cam_gray = cvCreateImage(cvGetSize(cam_image), IPL_DEPTH_8U, 1); 
	cam_frame = cvQueryFrameSafe(capture, sl_params);

	

	//IplImage* cam_gray = cvCreateImage(cvGetSize(cam_image), IPL_DEPTH_8U, 1); 
	cvSet(binary_image, cvScalar(0));
	cvCvtColor(cam_image,cam_gray,CV_RGB2GRAY);
	GaussianBlur(cam_gray,sl_params->cam_w,sl_params->cam_h);
	C_SobelMask(cam_gray,edge_image,binary_image,sl_params->cam_w,sl_params->cam_h);
	// Free allocated resources.
	cvReleaseImage(&cam_gray);
	return 0;
}