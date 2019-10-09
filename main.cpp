#include <dirent.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>

#define CLASS_NUMBER 0
#define INITIAL_VAL 0
#define PRECISION 2
#define COMMA ','
#define COLON ':'
#define SPACE ' '
#define PERCENT '%'
#define NEW_LINE '\n'
#define SEPERATOR "/"
#define SUFFIX ".csv"
#define DATASET "dataset"
#define LABELS "labels"
#define ACCURACY "Accuracy"

using namespace std;

typedef vector<vector<float> > Dataset;
typedef vector<vector<int> > Labels;
typedef vector<int> Prediction;
typedef vector<float> Scores;

enum Classifier
{
	BETHA_0,
	BETHA_1,
	BIAS
};

enum Feature
{
	LENGTH,
	WIDTH
};

enum Argument
{
	VALIDATION_DIR = 1,
	WEIGHT_VECTOR_DIR
};

void load_dataset(string filename, string dataset_directory,
		vector<Dataset>& datasets, vector<string>& files);
vector<float> get_new_float_row(string line);
void read_csv(Dataset& dataset, string filename);
void read_csv(Labels& labels, string filename);
string get_full_path(string directory_path, string filename);
bool ends_with(string first, string second);
vector<Dataset> get_datasets(vector<string>& files, string dataset_directory);
Scores get_scores(Dataset dataset, Dataset betha, 
		int instance_index, int number_of_classes);
int get_max_index(vector<float> list);
Prediction predict(Dataset dataset, Labels labels, 
		Dataset betha, int number_of_classes);
float get_score(Dataset dataset, Labels labels,
		Prediction label_prediction);
int get_final_class(vector<Prediction> label_predictions, int instance_index, 
		int number_of_classes, int number_of_classifier);
Prediction voter(Dataset dataset, vector<Prediction> label_predictions, 
		int number_of_classes, int number_of_classifier);
void predict_linearly(vector<Prediction>& label_predictions, 
		vector<Dataset> weight_vectors, Dataset dataset, Labels labels, 
		int number_of_classes, vector<string> files);
void predict_hard_voting(vector<Prediction> label_predictions, Dataset dataset, 
		Labels labels, int number_of_classes, int number_of_classifier);
void predict(string validation_dir, string weight_vector_dir);
void modify_command_argument(int counter, char const *arguments[]);

vector<float> get_new_float_row(string line)
{
	istringstream templine(line);
	string data;
	vector<float> row;

	while (getline(templine, data, COMMA)){
		row.push_back(atof(data.c_str()));
	}

	return row;
}

vector<int> get_new_int_row(string line)
{
	istringstream templine(line);
	string data;
	vector<int> row;

	while (getline(templine, data, COMMA)){
		row.push_back(atof(data.c_str()));
	}

	return row;
}

void read_csv(Dataset& dataset, string filename)
{
	ifstream file;
	file.open(filename);
	string line;
	getline(file, line, NEW_LINE);

	while (getline(file, line, NEW_LINE))
		dataset.push_back(get_new_float_row(line));

	file.close();
}

void read_csv(Labels& labels, string filename)
{
	ifstream file;
	file.open(filename);
	string line;
	getline(file, line, NEW_LINE);

	while (getline(file, line, NEW_LINE))
		labels.push_back(get_new_int_row(line));

	file.close();
}

string get_full_path(string directory_path, string filename)
{
	return (directory_path + filename);
}

bool ends_with(string first, string second)
{
	int first_LENGTH = first.size();
	int second_LENGTH = second.size();

	if (first_LENGTH < second_LENGTH)
		return false;

	if (first.substr(first_LENGTH - second_LENGTH,
			first_LENGTH - 1) == second)
		return true;

	return false;
}

void load_dataset(string filename, string dataset_directory,
		vector<Dataset>& datasets, vector<string>& files)
{
	files.push_back(filename);
	string full_path = get_full_path(dataset_directory, filename);
	Dataset dataset;
	read_csv(dataset, full_path);
	datasets.push_back(dataset);
}

vector<Dataset> get_datasets(vector<string>& files, string dataset_directory)
{
	vector<Dataset> datasets;
	DIR* directory;

	if ((directory = opendir(dataset_directory.c_str())) != nullptr)
	{
		struct dirent* entry;
		
		while ((entry = readdir(directory)) != nullptr)
		{
			string filename = string(entry->d_name);		
			if (ends_with(filename, SUFFIX))
				load_dataset(filename, dataset_directory,
						datasets, files);
		}
		closedir(directory);
	}
	else
		cerr << "Could not open directory..." << endl;

	return datasets;
}

Scores get_scores(Dataset dataset, Dataset betha, 
		int instance_index, int number_of_classes)
{
	Scores scores;

	for (int class_index = 0; class_index < number_of_classes; ++class_index)
	{
		float score = betha[class_index][BIAS] + 
				betha[class_index][BETHA_0] * dataset[instance_index][LENGTH] + 
				betha[class_index][BETHA_1] * dataset[instance_index][WIDTH];
		scores.push_back(score);
	}

	return scores;
}

int get_max_index(vector<float> list)
{
	float max = list[0];
	int max_index = 0;

	for (int i = 1; i < list.size(); ++i)
		if (list[i] > max)
		{
			max = list[i];
			max_index = i;
		}

	return max_index;
}

Prediction predict(Dataset dataset, Labels labels, 
		Dataset betha, int number_of_classes)
{
	Prediction label_prediction;

	for (int index = 0; index < dataset.size(); ++index)
	{
		Scores scores = get_scores(dataset, betha, index, number_of_classes);
		int class_number = get_max_index(scores);
		label_prediction.push_back(class_number);
	}

	return label_prediction;
}

float get_score(Dataset dataset, Labels labels,
		Prediction label_prediction)
{
	float satisfied = 0;

	for (int i = 0; i < labels.size(); ++i)
		if (labels[i][CLASS_NUMBER] == label_prediction[i])
			satisfied ++;

	float score = satisfied / label_prediction.size();	
	return score;
}

int get_final_class(vector<Prediction> label_predictions, int instance_index, 
		int number_of_classes, int number_of_classifier)
{
	vector<float> repetition(number_of_classes, INITIAL_VAL);

	for (int i = 0; i < number_of_classifier; ++i)
		repetition[label_predictions[i][instance_index]] ++;

	int final_class = get_max_index(repetition);
	return final_class;
}

Prediction voter(Dataset dataset, vector<Prediction> label_predictions, 
		int number_of_classes, int number_of_classifier)
{
	Prediction final_pred;
	
	for (int instance_index = 0; instance_index < dataset.size(); ++instance_index)
		final_pred.push_back(get_final_class(
				label_predictions, instance_index,
				number_of_classes,
				number_of_classifier));

	return final_pred;
}

void predict_linearly(vector<Prediction>& label_predictions,
		vector<Dataset> weight_vectors, Dataset dataset,
		Labels labels, int number_of_classes, vector<string> files)
{

	for (int i = 0; i < weight_vectors.size(); ++i)
	{
		Dataset betha = weight_vectors[i];
  		Prediction label_prediction = predict(dataset, labels, betha, 
  				number_of_classes);
  		float score = get_score(dataset, labels, label_prediction);
  		label_predictions.push_back(label_prediction);
  	}
}

void predict_hard_voting(vector<Prediction> label_predictions,
		Dataset dataset, Labels labels, int number_of_classes,
		int number_of_classifier)
{

	Prediction final_prediction = voter(dataset, label_predictions, 
			number_of_classes, number_of_classifier);
	float score = get_score(dataset, labels, final_prediction);
	cout << ACCURACY << COLON << SPACE << setprecision(PRECISION) << 
			fixed << (score * 100) << PERCENT << endl; 
}

void predict(string validation_dir, string weight_vector_dir)
{
	Dataset dataset;
	Labels labels;
	vector<string> files;
	vector<Prediction> label_predictions;
	vector<Dataset> weight_vectors = get_datasets(files, weight_vector_dir);
	int number_of_classifier = weight_vectors.size();
	int number_of_classes = weight_vectors[0].size();

	read_csv(dataset, validation_dir + DATASET + SUFFIX);
	read_csv(labels, validation_dir + LABELS + SUFFIX);

	predict_linearly(label_predictions, weight_vectors, dataset, labels, 
			number_of_classes, files);

	predict_hard_voting(label_predictions, dataset, labels, number_of_classes, 
			number_of_classifier);
}

int main(int argc, char const *argv[])
{
	string validation_dir = argv[VALIDATION_DIR];
	string weight_vector_dir = argv[WEIGHT_VECTOR_DIR];
	predict(validation_dir + SEPERATOR, 
			weight_vector_dir + SEPERATOR);
	return 0;
}