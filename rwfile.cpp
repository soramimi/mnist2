#include "rwfile.h"
#include <fcntl.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#define O_BINARY 0
#endif




bool readfile(const char *path, std::vector<char> *out)
{
	bool ok = false;
	out->clear();
	int fd = open(path, O_RDONLY | O_BINARY);
	if (fd != -1) {
		struct stat st;
		if (fstat(fd, &st) == 0 && st.st_size > 0) {
			out->resize(st.st_size);
			if (read(fd, out->data(), out->size()) == st.st_size) {
				ok = true;
			}
		}
		close(fd);
	}
	return ok;
}
