 @echo off
setlocal enabledelayedexpansion


set BM_VALUES=64 128
set BN_VALUES=64 128
set BK_VALUES=8 16
set TM_VALUES=4 8
set TN_VALUES=4 8


set OUTPUT_FILE=results.txt
echo BM BN BK TM TN GFLOPs > %OUTPUT_FILE%


for %%B in (%BM_VALUES%) do (
    for %%N in (%BN_VALUES%) do (
        for %%K in (%BK_VALUES%) do (
            for %%T in (%TM_VALUES%) do (
                for %%n in (%TN_VALUES%) do (
                    
                    echo #define BM %%B > params.h
                    echo #define BN %%N >> params.h
                    echo #define BK %%K >> params.h
                    echo #define TM %%T >> params.h
                    echo #define TN %%n >> params.h
                    type sgemm_template.cu >> params.h
                    
                    
                    nvcc -o sgemm 6_sgemmVectorize.cu
                    
                    
                    for /f "tokens=2 delims=: " %%a in ('sgemm') do (
                        set GFLOPS=%%a
                    )
                    
                    
                    echo %%B %%N %%K %%T %%n !GFLOPS! >> %OUTPUT_FILE%
                )
            )
        )
    )
)

echo Autotuning complete. Results saved to %OUTPUT_FILE%.
endlocal
pause
