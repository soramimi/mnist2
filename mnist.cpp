#include "mnist.h"
#include "rwfile.h"
#include <string.h>

Matrix::Matrix()
{
	d = std::make_shared<Data>();
}

Matrix::Matrix(const Matrix &r)
{
	d = std::make_shared<Data>();
	*d = *r.d;
}

void Matrix::make(size_t r, size_t c)
{
	d->rows = r;
	d->cols = c;
	d->vals.clear();
	d->vals.resize(d->rows * d->cols);
}

void Matrix::make(size_t r, size_t c, const std::initializer_list<Matrix::real_t> &list)
{
	d->rows = r;
	d->cols = c;
	d->vals.clear();
	d->vals = list;
}

void Matrix::make(size_t r, size_t c, const Matrix::real_t *p)
{
	d->rows = r;
	d->cols = c;
	d->vals.clear();
	d->vals.reserve(d->rows * d->cols);
	for (auto i = 0U; i < d->rows; i++) {
		for (auto j = 0U; j < d->cols; j++) {
			d->vals.push_back(*p);
			p++;
		}
	}
}

Matrix Matrix::transpose() const
{
	Matrix out;
	out.make(d->cols, d->rows);
	for (auto r = 0U; r < d->rows; r++) {
		for (auto c = 0U; c < d->cols; c++) {
			out.at(c, r) = at(r, c);
		}
	}
	return out;
}

Matrix Matrix::dot(const Matrix &other) const
{
	Matrix out;
	Matrix const &a = *this;
	Matrix const &b = other;
	auto n = a.d->cols;
	if (n == b.d->rows) {
		auto nrow = a.d->rows;
		auto ncol = b.d->cols;
		out.make(nrow, ncol);
		for (auto col = 0U; col < ncol; col++) {
			for (auto row = 0U; row < nrow; row++) {
				for (auto i = 0U; i < n; i++) {
					out.at(row, col) += a.at(row, i) * b.at(i , col);
				}
			}
		}
	}
	return out;
}

Matrix Matrix::add(const Matrix &other) const
{
	Matrix out(*this);
	size_t n = std::min(d->vals.size(), other.d->vals.size());
	for (size_t i = 0; i < n; i++) {
		out.d->vals[i] += other.d->vals[i];
	}
	return out;
}

Matrix Matrix::sub(const Matrix &other) const
{
	Matrix out(*this);
	size_t n = std::min(d->vals.size(), other.d->vals.size());
	for (size_t i = 0; i < n; i++) {
		out.d->vals[i] -= other.d->vals[i];
	}
	return out;
}

Matrix Matrix::mul(const Matrix &other) const
{
	Matrix out(*this);
	size_t n = std::min(d->vals.size(), other.d->vals.size());
	for (size_t i = 0; i < n; i++) {
		out.d->vals[i] *= other.d->vals[i];
	}
	return out;
}

Matrix Matrix::mul(Matrix::real_t t) const
{
	Matrix out(*this);
	size_t n = d->vals.size();
	for (size_t i = 0; i < n; i++) {
		out.d->vals[i] *= t;
	}
	return out;
}

Matrix Matrix::div(Matrix::real_t t) const
{
	Matrix out(*this);
	size_t n = d->vals.size();
	for (size_t i = 0; i < n; i++) {
		out.d->vals[i] /= t;
	}
	return out;
}

Matrix Matrix::sum() const
{
	Matrix out;
	out.make(1, d->cols);
	for (auto r = 0U; r < d->rows; r++) {
		for (auto c = 0U; c < d->cols; c++) {
			out.at(0, c) += at(r, c);
		}
	}
	return out;
}

Matrix Matrix::sigmoid() const
{
	Matrix out;
	out.make(d->rows, d->cols);
	size_t n = d->vals.size();
	for (size_t i = 0; i < n; i++) {
		out.d->vals[i] = sigmoid(d->vals[i]);
	}
	return out;
}

Matrix Matrix::sigmoid_grad() const
{
	Matrix out;
	out.make(d->rows, d->cols);
	size_t n = d->vals.size();
	for (size_t i = 0; i < n; i++) {
		real_t v = sigmoid(d->vals[i]);
		out.d->vals[i] = (1 - v) * v;
	}
	return out;
}

Matrix Matrix::softmax() const
{
	Matrix out;
	out.make(d->rows, d->cols);
	for (size_t r = 0; r < d->rows; r++) {
		real_t c = 0;
		for (size_t i = 0; i < d->cols; i++) {
			c = std::max(c, at(r, i));
		}
		std::vector<real_t> exp_a(d->cols);
		real_t sum_exp_a = 0;
		for (size_t i = 0; i < d->cols; i++) {
			real_t v = exp(at(r, i) - c);
			exp_a[i] = v;
			sum_exp_a += v;
		}
		for (size_t i = 0; i < d->cols; i++) {
			out.at(r, i) = exp_a[i] / sum_exp_a;
		}
	}
	return out;
}

void Matrix::add_rows(const Matrix &other)
{
	if (d->cols == 0 && d->rows == 0) {
		make(0, other.d->cols);
	}
	if (d->cols == other.d->cols) {
		d->rows += other.d->rows;
		d->vals.insert(d->vals.end(), other.d->vals.begin(), other.d->vals.end());
	}
}

//

bool mnist::DataSet::image_to_matrix(int index, Matrix *out) const
{
	if (index < (int)data.images.size()) {
		int n = data.rows * data.cols;
		auto &image = data.images[index];
		out->make(1, data.rows * data.cols);
		for (int i = 0; i < n; i++) {
			out->data()[i] = image[i] / 255.0;
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
		out->data()[i] = (i == v);
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

