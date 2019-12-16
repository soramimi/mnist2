#include "mnist.h"
#include "rwfile.h"
#include <string.h>

bool mnist::DataSet::image_to_matrix(int index, Matrix *out) const
{
	if (index < (int)data.images.size()) {
		int n = data.rows * data.cols;
		auto &image = data.images[index];
		out->make(1, data.rows * data.cols);
		for (int i = 0; i < n; i++) {
			out->array[i] = image[i] / 255.0;
		}
		return true;
	}
	return false;
}

int mnist::DataSet::label(int index) const
{
	if (index < (int)data.labels.size()) {
		return data.labels[index];
	}
	return -1;
}

void mnist::DataSet::label_to_matrix(int index, Matrix *out) const
{
	out->make(1, 10);
	int v = label(index);
	for (int i = 0; i < 10; i++) {
		out->array[i] = (i == v);
	}
}



bool mnist::DataSet::load(const char *labels_path, const char *images_path)
{
	mnist::DataSet *out = this;
	out->data = {};

	bool ok = false;
	auto Read32BE = [](void const *p){
		uint8_t const *q = (uint8_t const *)p;
		return (q[0] << 24) | (q[1] << 16) | (q[2] << 8) | q[3];
	};
	size_t labels_count = 0;
	std::vector<char> labels_data;
	readfile(labels_path, &labels_data);
	if (labels_data.size() >= 8) {
		char const *begin = labels_data.data();
		char const *end = begin + labels_data.size();
		uint32_t sig = Read32BE(begin);
		if (sig == 0x00000801) {
			labels_count = Read32BE(begin + 4);
			labels_count = std::min(labels_count, size_t(end - begin - 8));
			out->data.labels.resize(labels_count);
		}
	}
	std::vector<char> images_data;
	readfile(images_path, &images_data);
	if (images_data.size() >= 16) {
		char const *begin = images_data.data();
		char const *end = begin + images_data.size();
		uint32_t sig = Read32BE(begin);
		if (sig == 0x00000803) {
			size_t count = Read32BE(begin + 4);
			out->data.rows = Read32BE(begin + 8);
			out->data.cols = Read32BE(begin + 12);
			count = std::min(count, size_t(end - begin - 16) / (out->data.cols * out->data.rows));
			count = std::min(count, labels_count);
			out->data.images.resize(count);
			for (size_t i = 0; i < count; i++) {
				out->data.images[i].resize(out->data.rows * out->data.cols);
				uint8_t *dst = out->data.images[i].data();
				uint8_t const *src = (uint8_t const *)begin + 16 + out->data.rows * out->data.cols * i;
				memcpy(dst, src, out->data.rows * out->data.cols);
			}
			memcpy(out->data.labels.data(), labels_data.data() + 8, count);
			out->data.count = count;
			ok = true;
		}
	}
	return ok;
}
