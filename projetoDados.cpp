#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp" 
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, const char** argv) {

	VideoCapture capture("../data/videos/videodados.wmv");

	if (!capture.isOpened()) {
		return -1;
	}

	Mat frame, clone, segmentacao;
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;
	namedWindow("Vídeo Dados");

	while (true) {

		if (!capture.read(frame)) {
			break;
		}

		clone = frame.clone();
		cvtColor(frame, segmentacao, CV_BGR2GRAY);
		threshold(segmentacao, segmentacao, 245, 255, CV_THRESH_BINARY_INV);

		Mat element = getStructuringElement(MORPH_CROSS, Size(7, 7));
		morphologyEx(segmentacao, segmentacao, MORPH_OPEN, element);

		//Busca por contornos
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> diceHierarchy;
		cv::findContours(segmentacao,
			contours, diceHierarchy,
			CV_RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE
		);

		for (int i = 0; i < contours.size(); ++i)
		{
			// Cria um retangulo referente ao blob
			Rect boundsRect = boundingRect(Mat(contours[i]));
			//rectangle(frame, Rect(boundsRect.x, boundsRect.y, boundsRect.width, boundsRect.height), Scalar(0, 0, 255));

			//Cria um Mat para o blob da imagem original
			Mat blob(Mat(clone, boundsRect));

			cvtColor(blob, blob, CV_BGR2GRAY);
			threshold(blob, blob, 127, 255, THRESH_BINARY);

			//Usa o SimpleBlobDetector para contar os pontos do dado
			std::vector<cv::KeyPoint> keypoints;
			cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create();
			blobDetector->detect(blob, keypoints);

			//Imprime no cento do dado a resposta
			Point center_of_rect = (boundsRect.br() + boundsRect.tl()) * 0.5;
			putText(frame, to_string(keypoints.size()), center_of_rect, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 125, 255), 4);
		}

		if (waitKey(delay) >= 0)
			break;

		imshow("Vídeo Dados", frame);
	}

	capture.release();

	return 0;
}