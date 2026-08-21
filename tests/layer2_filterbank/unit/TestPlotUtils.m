classdef TestPlotUtils < matlab.unittest.TestCase
%TESTPLOTUTILS  Smoke tests for FFT plotting utilities.
%
%   Covers: plotfft, plotfftreal
%
%   Both functions produce frequency-domain magnitude plots.  These tests
%   verify that they execute without error and accept expected input sizes;
%   they do not validate visual appearance.

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            % Suppress all figure windows during test run.
            set(0, 'DefaultFigureVisible', 'off');
        end
    end

    methods (TestClassTeardown)
        function teardownClass(~)
            close all;
            set(0, 'DefaultFigureVisible', 'on');
        end
    end

    % ── plotfft ───────────────────────────────────────────────────────────
    methods (Test)

        function testPlotfftRunsWithVector(tc)
            % plotfft must run without error for a full-length FFT vector.
            L = 64;
            f = randn(L, 1);
            F = fft(f);
            tc.verifyWarningFree(@() plotfft(F), ...
                'plotfft: must run without warnings for a length-L FFT vector.');
            close all;
        end

        function testPlotfftRunsWithFs(tc)
            % plotfft must accept an optional sampling-rate argument.
            L = 64;
            fs = 8000;
            F = fft(randn(L, 1));
            try
                plotfft(F, fs);
                close all;
                tc.verifyTrue(true, 'plotfft with fs ran without error.');
            catch ME
                tc.verifyFail(sprintf('plotfft(F,fs) threw: %s', ME.message));
            end
        end

        function testPlotfftRunsWithDynrange(tc)
            % plotfft must accept a dynamic-range limiting argument.
            L = 64;
            F = fft(randn(L, 1));
            try
                plotfft(F, 8000, 60);
                close all;
                tc.verifyTrue(true, 'plotfft with dynrange ran without error.');
            catch ME
                tc.verifyFail(sprintf('plotfft(F,fs,dynrange) threw: %s', ME.message));
            end
        end

        function testPlotfftCreatesAxes(tc)
            % After calling plotfft an axes object should exist.
            L = 128;
            F = fft(randn(L, 1));
            figure('Visible', 'off');
            plotfft(F);
            ax = gca;
            tc.verifyTrue(ishandle(ax) && strcmp(get(ax,'Type'),'axes'), ...
                'plotfft: must produce a valid axes handle.');
            close all;
        end

    end

    % ── plotfftreal ───────────────────────────────────────────────────────
    methods (Test)

        function testPlotfftrealRunsWithVector(tc)
            % plotfftreal must run without error for a positive-frequency vector.
            L = 64;
            f = randn(L, 1);
            F = fftreal(f);         % length floor(L/2)+1
            tc.verifyWarningFree(@() plotfftreal(F), ...
                'plotfftreal: must run without warnings for a fftreal vector.');
            close all;
        end

        function testPlotfftrealRunsWithFs(tc)
            % plotfftreal must accept an optional sampling-rate argument.
            L = 64;
            fs = 8000;
            F = fftreal(randn(L, 1));
            try
                plotfftreal(F, fs);
                close all;
                tc.verifyTrue(true, 'plotfftreal with fs ran without error.');
            catch ME
                tc.verifyFail(sprintf('plotfftreal(F,fs) threw: %s', ME.message));
            end
        end

        function testPlotfftrealRunsWithDynrange(tc)
            % plotfftreal must accept a dynamic-range argument.
            L = 64;
            F = fftreal(randn(L, 1));
            try
                plotfftreal(F, 8000, 60);
                close all;
                tc.verifyTrue(true, 'plotfftreal with dynrange ran without error.');
            catch ME
                tc.verifyFail(sprintf('plotfftreal(F,fs,dynrange) threw: %s', ME.message));
            end
        end

        function testPlotfftrealCreatesAxes(tc)
            % After calling plotfftreal an axes object should exist.
            L = 128;
            F = fftreal(randn(L, 1));
            figure('Visible', 'off');
            plotfftreal(F);
            ax = gca;
            tc.verifyTrue(ishandle(ax) && strcmp(get(ax,'Type'),'axes'), ...
                'plotfftreal: must produce a valid axes handle.');
            close all;
        end

        function testPlotfftrealVsPlotfftConsistency(tc)
            % plotfft and plotfftreal must both accept the same signal and
            % complete without error: no crash for either full or half spectrum.
            L = 128;
            f = randn(L, 1);
            F_full = fft(f);
            F_real = fftreal(f);
            try
                plotfft(F_full);
                close all;
                plotfftreal(F_real);
                close all;
                tc.verifyTrue(true, 'Both plotfft and plotfftreal ran without error.');
            catch ME
                tc.verifyFail(sprintf('Plotting threw: %s', ME.message));
            end
        end

    end

end
