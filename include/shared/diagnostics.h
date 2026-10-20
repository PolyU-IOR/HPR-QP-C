#ifndef HPRQP_DIAGNOSTICS_H
#define HPRQP_DIAGNOSTICS_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Feature diagnostics are independent of the ordinary HPR progress table. */
static inline int hprqp_diagnostics_enabled(void)
{
    const char *value = getenv("HPRQP_DIAGNOSTICS");
    return value != NULL && strcmp(value, "1") == 0;
}

static inline void hprqp_diagnostic(const char *format, ...)
{
    va_list args;
    if (!hprqp_diagnostics_enabled()) return;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

#endif
