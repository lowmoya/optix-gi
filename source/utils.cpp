
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"

char * readFile(char const * path, int & length)
{
	FILE * file = fopen(path, "r");
	if (!file) {
		length = 0;
		return nullptr;
	}

	fseek(file, 0, SEEK_END);
	length = ftell(file);
	rewind(file);

	char * buffer = (char *)malloc(length + 1);
	fread(buffer, 1, length, file);
	buffer[length] = '\0';

	fclose(file);
	return buffer;
}