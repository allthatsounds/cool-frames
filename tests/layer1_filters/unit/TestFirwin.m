classdef TestFirwin < matlab.unittest.TestCase
%TESTFIRWIN  Unit tests for firwin, freqwin, freqwavelet.
%
%   firwin(name, M)  — symmetric FIR window of length M
%   freqwin(name, L, bw)  — bandpass frequency window, peak-normalised
%   freqwavelet(name, L)  — mother wavelet in frequency domain, peak-normalised


    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── firwin: output length ────────────────────────────────────────────────
    methods (Test)

        function testFirwinOutputLength(tc)
            for M = [16, 32, 64, 128]
                w = firwin('hann', M);
                tc.verifyEqual(numel(w), M, ...
                    sprintf('firwin hann: expected length %d', M));
            end
        end

        function testFirwinOutputLengthOdd(tc)
            for M = [15, 33, 65]
                w = firwin('hann', M);
                tc.verifyEqual(numel(w), M, ...
                    sprintf('firwin hann odd: expected length %d', M));
            end
        end

        function testFirwinSineLength(tc)
            w = firwin('sine', 64);
            tc.verifyEqual(numel(w), 64);
        end

        function testFirwinRectLength(tc)
            w = firwin('rect', 32);
            tc.verifyEqual(numel(w), 32);
        end

    end

    % ── firwin: symmetry (whole-point even) ──────────────────────────────────
    methods (Test)

        function testFirwinHannWPESymmetry(tc)
            % w(k) = w(M+2-k) for k = 2..M  (whole-point even)
            M = 64;
            w = firwin('hann', M);
            tc.verifyEqual(w(2:end), flipud(w(2:end)), 'AbsTol', 1e-14, ...
                'hann: not whole-point even symmetric');
        end

        function testFirwinSineWPESymmetry(tc)
            M = 64;
            w = firwin('sine', M);
            tc.verifyEqual(w(2:end), flipud(w(2:end)), 'AbsTol', 1e-14, ...
                'sine: not whole-point even symmetric');
        end

        function testFirwinRectWPESymmetry(tc)
            M = 64;
            w = firwin('rect', M);
            tc.verifyEqual(w(2:end), flipud(w(2:end)), 'AbsTol', 1e-14, ...
                'rect: not whole-point even symmetric');
        end

        function testFirwinTriaWPESymmetry(tc)
            M = 64;
            w = firwin('tria', M);
            tc.verifyEqual(w(2:end), flipud(w(2:end)), 'AbsTol', 1e-14, ...
                'tria: not whole-point even symmetric');
        end

    end

    % ── firwin: non-negativity ───────────────────────────────────────────────
    methods (Test)

        function testFirwinHannNonNegative(tc)
            w = firwin('hann', 64);
            tc.verifyGreaterThanOrEqual(min(w), -1e-14, ...
                'hann window should be non-negative');
        end

        function testFirwinSineNonNegative(tc)
            w = firwin('sine', 64);
            tc.verifyGreaterThanOrEqual(min(w), -1e-14);
        end

        function testFirwinRectIsOnesVector(tc)
            % For even M, the sampling grid includes x = -M/2/M = -0.5, and
            % abs(-0.5) < 0.5 is false, so that one element is 0 (the docstring
            % calls it "(Almost) rectangular").  Use odd M to get a true all-ones
            % vector: odd grids are symmetric around 0 with |x| < 0.5 for all pts.
            M = 33;
            w = firwin('rect', M);
            tc.verifyEqual(w, ones(M, 1), 'AbsTol', 1e-14, ...
                'rect window (odd M) should be all-ones');
        end

    end

    % ── firwin: partition of unity ───────────────────────────────────────────
    methods (Test)

        function testFirwinHannPU(tc)
            % hann: w + fftshift(w) = ones  (for even M)
            M = 64;
            w = firwin('hann', M);
            tc.verifyEqual(w + fftshift(w), ones(M, 1), 'AbsTol', 1e-14, ...
                'hann: partition of unity w + fftshift(w) = 1 failed');
        end

        function testFirwinHannPUMultipleLengths(tc)
            for M = [32, 64, 128, 256]
                w = firwin('hann', M);
                tc.verifyEqual(w + fftshift(w), ones(M, 1), 'AbsTol', 1e-14, ...
                    sprintf('hann PU failed for M=%d', M));
            end
        end

        function testFirwinSineTightFrame(tc)
            % sine = sqrt(hann): sine.^2 + fftshift(sine.^2) = ones  (tight frame)
            M = 64;
            w = firwin('sine', M);
            tc.verifyEqual(w.^2 + fftshift(w.^2), ones(M, 1), 'AbsTol', 1e-14, ...
                'sine: tight frame condition sine^2 + fftshift(sine^2) = 1 failed');
        end

        function testFirwinSineTightFrameMultipleLengths(tc)
            for M = [32, 64, 128]
                w = firwin('sine', M);
                tc.verifyEqual(w.^2 + fftshift(w.^2), ones(M, 1), 'AbsTol', 1e-14, ...
                    sprintf('sine tight frame failed for M=%d', M));
            end
        end

    end

    % ── firwin: peak at DC (index 1) ─────────────────────────────────────────
    methods (Test)

        function testFirwinHannPeakAtDC(tc)
            % firwin returns zero-phase window: max at index 1
            w = firwin('hann', 64);
            [~, idx] = max(w);
            tc.verifyEqual(idx, 1, 'hann: peak should be at index 1 (DC)');
        end

        function testFirwinSinePeakAtDC(tc)
            w = firwin('sine', 64);
            [~, idx] = max(w);
            tc.verifyEqual(idx, 1, 'sine: peak should be at index 1 (DC)');
        end

    end

    % ── freqwin: output ──────────────────────────────────────────────────────
    methods (Test)

        function testFreqwinOutputLength(tc)
            L  = 128;
            bw = 0.1;
            w  = freqwin('gauss', L, bw);
            tc.verifyEqual(numel(w), L, 'freqwin gauss: wrong output length');
        end

        function testFreqwinPeakNormalized(tc)
            % freqwin default: peak-normalized, max|w| = 1
            L  = 256;
            bw = 0.05;
            w  = freqwin('gauss', L, bw);
            tc.verifyEqual(max(abs(w)), 1, 'AbsTol', 1e-12, ...
                'freqwin: should be peak-normalized');
        end

        function testFreqwinGammatoneLength(tc)
            w = freqwin('gammatone', 128, 0.1);
            tc.verifyEqual(numel(w), 128);
        end

        function testFreqwinButterworthLength(tc)
            w = freqwin('butterworth', 128, 0.1);
            tc.verifyEqual(numel(w), 128);
        end

        function testFreqwinNonNegative(tc)
            % Frequency-domain windows are non-negative (they are magnitude shapes)
            w = freqwin('gauss', 128, 0.1);
            tc.verifyGreaterThanOrEqual(min(w), -1e-12, ...
                'freqwin gauss: should be non-negative');
        end

    end

    % ── freqwavelet: output ──────────────────────────────────────────────────
    methods (Test)

        function testFreqwaveletOutputLength(tc)
            L = 256;
            H = freqwavelet('cauchy', L);
            tc.verifyEqual(numel(H), L, 'freqwavelet: wrong output length');
        end

        function testFreqwaveletPeakNormalized(tc)
            % freqwavelet defaults to 'null' normalization (no scaling).
            % Pass 'peak' explicitly to obtain max|H| = 1.
            H = freqwavelet('cauchy', 256, 'peak');
            tc.verifyEqual(max(abs(H)), 1, 'AbsTol', 1e-12, ...
                'freqwavelet cauchy with peak norm: max|H| should be 1');
        end

        function testFreqwaveletMorletLength(tc)
            H = freqwavelet('morlet', 256);
            tc.verifyEqual(numel(H), 256);
        end

        function testFreqwaveletMorseLength(tc)
            H = freqwavelet('morse', 256);
            tc.verifyEqual(numel(H), 256);
        end

        function testFreqwaveletFiniteValues(tc)
            % No NaN or Inf for standard parameters
            for name = {'cauchy', 'morlet', 'morse'}
                H = freqwavelet(name{1}, 256);
                tc.verifyTrue(all(isfinite(H)), ...
                    sprintf('freqwavelet %s: non-finite values', name{1}));
            end
        end

    end

end
