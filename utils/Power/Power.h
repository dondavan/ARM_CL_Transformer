//Ehsan Power:
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define IN 0
#define OUT 1

#define LOW 0
#define HIGH 1

//Khadas
//#define POUT 432
//RockPi
#define POUT 157 /* P1-07 */

#define BUFFER_MAX 3
#define DIRECTION_MAX 35
#define VALUE_MAX 30

static const char s_directions_str[] = "in\0out";
static const char s_values_str[] = "01";

namespace power
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
int GPIOExport(int pin)
{
    char    buffer[BUFFER_MAX];
    ssize_t bytes_written;
    int     fd;

    fd = open("/sys/class/gpio/export", O_WRONLY);
    if(-1 == fd)
    {
        fprintf(stderr, "Failed to open export for writing!\n");
        return (-1);
    }

    bytes_written = snprintf(buffer, BUFFER_MAX, "%d", pin);
    int ret = write(fd, buffer, bytes_written);
    close(fd);
    return (0);
}

int GPIODirection(int pin, int dir)
{
    char path[DIRECTION_MAX];
    int  fd;

    snprintf(path, DIRECTION_MAX, "/sys/class/gpio/gpio%d/direction", pin);
    fd = open(path, O_WRONLY);
    if(-1 == fd)
    {
        fprintf(stderr, "Failed to open gpio direction for writing!\n");
        return (-1);
    }

    if(-1 == write(fd, &s_directions_str[IN == dir ? 0 : 3], IN == dir ? 2 : 3))
    {
        fprintf(stderr, "Failed to set direction!\n");
        return (-1);
    }

    close(fd);
    return (0);
}

int GPIOWrite(int pin, int value)
{
    char path[VALUE_MAX];
    int  fd;

    snprintf(path, VALUE_MAX, "/sys/class/gpio/gpio%d/value", pin);
    fd = open(path, O_WRONLY);
    if(-1 == fd)
    {
        fprintf(stderr, "Failed to open gpio value for writing!\n");
        return (-1);
    }

    if(1 != write(fd, &s_values_str[LOW == value ? 0 : 1], 1))
    {
        fprintf(stderr, "Failed to write value!\n");
        return (-1);
    }

    close(fd);
    return (0);
}

int GPIOUnexport(int pin)
{
    char    buffer[BUFFER_MAX];
    ssize_t bytes_written;
    int     fd;

    fd = open("/sys/class/gpio/unexport", O_WRONLY);
    if(-1 == fd)
    {
        fprintf(stderr, "Failed to open unexport for writing!\n");
        return (-1);
    }

    bytes_written = snprintf(buffer, BUFFER_MAX, "%d", pin);
    int ret = write(fd, buffer, bytes_written);
    close(fd);
    return (0);
}
#pragma GCC diagnostic pop

}// namespace power