/* env.h - environment recorder. The harness writes {"type":"env",...} as the first JSON line and a
 * short human-readable header; tests can query the same facts. */
#ifndef T_ENV_H
#define T_ENV_H
#include <stdio.h>
#include <stdint.h>

typedef struct t_env {
    unsigned long os_major, os_minor, os_build, os_ubr;  /* RtlGetVersion; UBR from registry (0 if unknown) */
    char     os_product[128];                            /* registry ProductName / DisplayVersion, best effort */
    const char *process_arch;                            /* "x64" / "x86" */
    int      wow64;                                      /* 32-bit process on a 64-bit OS */
    unsigned short native_machine;                       /* IMAGE_FILE_MACHINE_* (IsWow64Process2) or 0 */
    char     wine[64];                                   /* wine_get_version() or "" */
    int      ncpu;                                       /* logical CPUs in process affinity */
    unsigned long long affinity;                         /* process affinity mask (group 0) */
    unsigned short active_groups;
    unsigned long long qpc_freq;
    int      api_xstate_present;                         /* GetEnabledXStateFeatures exported */
    unsigned long long api_xstate_features;              /* GetEnabledXStateFeatures() */
    unsigned char pf[64];                                /* IsProcessorFeaturePresent(0..63) */
} t_env;

const t_env *t_env_get(void);                            /* collected once */
void t_env_print(FILE *f);                               /* human header (ENV lines) */
void t_env_write_json(FILE *f, const char *exe, const char *version);  /* one {"type":"env"} line */
#endif
