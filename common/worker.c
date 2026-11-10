/* worker.c - t_run_worker: a case helper thread with a bounded wait that can never contaminate later cases. */
#include "t.h"

typedef struct { t_worker_fn fn; void *arg; } worker_block;

static DWORD WINAPI worker_tramp(void *p) {
    worker_block *b = (worker_block *)p;
    b->fn(b->arg);
    return 0;
}

int t_run_worker(t_worker_fn fn, void *arg, SIZE_T stack, DWORD timeout_ms, DWORD *code) {
    /* heap block, never freed on the retire path: a worker that is still running keeps a valid pointer */
    worker_block *b = (worker_block *)HeapAlloc(GetProcessHeap(), 0, sizeof *b);
    if (!b) { SetLastError(ERROR_NOT_ENOUGH_MEMORY); return -1; }
    b->fn = fn; b->arg = arg;
    HANDLE h = CreateThread(NULL, stack, worker_tramp, b, 0, NULL);
    if (!h) { DWORD e = GetLastError(); HeapFree(GetProcessHeap(), 0, b); SetLastError(e); return -1; }
    DWORD w = WaitForSingleObject(h, timeout_ms);
    if (w != WAIT_OBJECT_0) {
        DWORD e = (w == WAIT_FAILED) ? GetLastError() : 0;
        char why[240];
        if (w == WAIT_TIMEOUT)
            snprintf(why, sizeof why, "worker thread did not finish within %lu ms (WAIT_TIMEOUT); process retired, no TerminateThread",
                     (unsigned long)timeout_ms);
        else
            snprintf(why, sizeof why, "worker thread wait failed (result 0x%lx, GetLastError %lu); process retired",
                     (unsigned long)w, (unsigned long)e);
        t_retire_process(why);
    }
    if (code) GetExitCodeThread(h, code);
    CloseHandle(h);
    HeapFree(GetProcessHeap(), 0, b);
    return 0;
}
