classdef TestAudScaleUtils < matlab.unittest.TestCase
%TESTAUDSCALEUTILS  Unit tests for auditory-scale conversion utilities and
%                   signal processing helpers.
%
%   Covers: erbtofreq, freqtoerb, pfilt, setnorm
%
%   erbtofreq(erb)       -- ERB number → Hz (wrapper around audtofreq)
%   freqtoerb(freq)      -- Hz → ERB number (wrapper around freqtoaud)
%   pfilt(f, g)          -- apply filter g with periodic boundary conditions
%   setnorm(f, ...)      -- set signal norm to a specified value

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── erbtofreq ─────────────────────────────────────────────────────────
    methods (Test)

        function testErbtofreqScalar(tc)
            % 0 ERBs must map to 0 Hz.
            tc.verifyEqual(erbtofreq(0), 0, 'AbsTol', 1, ...
                'erbtofreq: 0 ERB should map to ~0 Hz.');
        end

        function testErbtofreqPositiveOutput(tc)
            % Positive ERB values must give positive frequencies.
            erbs = [1, 5, 10, 20, 30];
            freqs = erbtofreq(erbs);
            tc.verifyTrue(all(freqs > 0), ...
                'erbtofreq: positive ERB values must yield positive Hz.');
        end

        function testErbtofreqMonotonicallyIncreasing(tc)
            % ERB scale is monotonically increasing in frequency.
            erbs = 0:0.5:35;
            freqs = erbtofreq(erbs);
            tc.verifyTrue(all(diff(freqs) > 0), ...
                'erbtofreq: output must be strictly monotonically increasing.');
        end

        function testErbtofreqVectorLength(tc)
            % Output must have the same number of elements as input.
            erbs = linspace(1, 30, 50);
            freqs = erbtofreq(erbs);
            tc.verifyEqual(numel(freqs), numel(erbs), ...
                'erbtofreq: output length must match input length.');
        end

        function testErbtofreqInverseRoundtrip(tc)
            % erbtofreq(freqtoerb(f)) ≈ f for positive Hz values.
            freqs_in = [100, 500, 1000, 2000, 4000, 8000];
            freqs_out = erbtofreq(freqtoerb(freqs_in));
            rel_err = norm(freqs_out - freqs_in) / norm(freqs_in);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'erbtofreq(freqtoerb(f)) must recover f (round-trip consistency).');
        end

    end

    % ── freqtoerb ─────────────────────────────────────────────────────────
    methods (Test)

        function testFreqtoerbScalar(tc)
            % 1000 Hz maps to a specific known ERB value (approximately 15.7).
            erb_1k = freqtoerb(1000);
            tc.verifyGreaterThan(erb_1k, 10, ...
                'freqtoerb: 1000 Hz must map to ERB > 10.');
            tc.verifyLessThan(erb_1k, 25, ...
                'freqtoerb: 1000 Hz must map to ERB < 25.');
        end

        function testFreqtoerbMonotonicallyIncreasing(tc)
            % Frequency values must map monotonically to ERB values.
            freqs = [100, 500, 1000, 2000, 4000, 8000];
            erbs = freqtoerb(freqs);
            tc.verifyTrue(all(diff(erbs) > 0), ...
                'freqtoerb: ERB values must be strictly increasing with frequency.');
        end

        function testFreqtoerbVectorLength(tc)
            % Output must have the same number of elements as input.
            freqs = linspace(100, 8000, 40);
            erbs = freqtoerb(freqs);
            tc.verifyEqual(numel(erbs), numel(freqs), ...
                'freqtoerb: output length must match input length.');
        end

        function testFreqtoerbPositiveOutput(tc)
            % Positive Hz values must give positive ERB numbers.
            freqs = [250, 1000, 4000];
            erbs = freqtoerb(freqs);
            tc.verifyTrue(all(erbs > 0), ...
                'freqtoerb: positive Hz must yield positive ERB numbers.');
        end

        function testFreqtoerbInverseRoundtrip(tc)
            % freqtoerb(erbtofreq(e)) ≈ e for positive ERB values.
            erbs_in = [5, 10, 15, 20, 25];
            erbs_out = freqtoerb(erbtofreq(erbs_in));
            rel_err = norm(erbs_out - erbs_in) / norm(erbs_in);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'freqtoerb(erbtofreq(e)) must recover e (round-trip consistency).');
        end

    end

    % ── pfilt ─────────────────────────────────────────────────────────────
    methods (Test)

        function testPfiltOutputLengthNoSubsampling(tc)
            % With a=1, output length must equal input length.
            L = 128;
            f = randn(L, 1);
            g = firwin('hann', 16);
            c = pfilt(f, g);
            tc.verifyEqual(numel(c), L, ...
                'pfilt: output length must equal input length when a=1.');
        end

        function testPfiltOutputLengthSubsampled(tc)
            % With subsampling a, output length must be L/a.
            L = 128;
            a = 4;
            f = randn(L, 1);
            g = firwin('hann', 16);
            c = pfilt(f, g, a);
            tc.verifyEqual(numel(c), L/a, ...
                sprintf('pfilt: output length must be L/a = %d with a=%d.', L/a, a));
        end

        function testPfiltImpulseResponseIdentity(tc)
            % Filtering an impulse with a Dirac filter should return the impulse.
            L = 64;
            f = [1; zeros(L-1, 1)];
            g = [1; zeros(L-1, 1)];      % Dirac in time domain
            c = pfilt(f, g);
            tc.verifyEqual(c(1), 1, 'AbsTol', 1e-12, ...
                'pfilt: filtering impulse with Dirac must return impulse at DC.');
        end

        function testPfiltOutputIsNumeric(tc)
            % pfilt must return a numeric array.
            L = 64;
            f = randn(L, 1);
            g = firwin('hann', 8);
            c = pfilt(f, g);
            tc.verifyTrue(isnumeric(c), 'pfilt: output must be numeric.');
        end

        function testPfiltPeriodicBoundaryConsistency(tc)
            % pfilt with a full-length DFT filter must equal pointwise
            % multiplication in the frequency domain.
            L = 64;
            f = randn(L, 1);
            G = randn(L, 1) + 1i * randn(L, 1);
            g_struct.h = G;
            g_struct.foff = 0;
            g_struct.L = L;
            % Compute via pfilt
            c_pfilt = pfilt(f, g_struct.h);
            % Compute via direct DFT multiplication
            c_direct = ifft(fft(f) .* fft(G));
            rel_err = norm(c_pfilt - c_direct) / (norm(c_direct) + eps);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'pfilt: full-length H filter must match direct DFT convolution.');
        end

    end

    % ── setnorm ───────────────────────────────────────────────────────────
    methods (Test)

        function testSetnormL2(tc)
            % After setnorm(f,'2'), norm(f) must equal 1.
            f = randn(64, 1);
            fn = setnorm(f, '2');
            tc.verifyEqual(norm(fn), 1, 'AbsTol', 1e-12, ...
                'setnorm(''2''): resulting l2 norm must be 1.');
        end

        function testSetnormEnergy(tc)
            % 'energy' is equivalent to '2'.
            f = randn(64, 1);
            fn = setnorm(f, 'energy');
            tc.verifyEqual(norm(fn), 1, 'AbsTol', 1e-12, ...
                'setnorm(''energy''): resulting l2 norm must be 1.');
        end

        function testSetnormL1(tc)
            % After setnorm(f,'1'), sum(abs(f)) must equal 1.
            f = randn(64, 1);
            fn = setnorm(f, '1');
            tc.verifyEqual(sum(abs(fn)), 1, 'AbsTol', 1e-12, ...
                'setnorm(''1''): resulting l1 norm must be 1.');
        end

        function testSetnormInf(tc)
            % After setnorm(f,'inf'), max(abs(f)) must equal 1.
            f = randn(64, 1);
            fn = setnorm(f, 'inf');
            tc.verifyEqual(max(abs(fn)), 1, 'AbsTol', 1e-12, ...
                'setnorm(''inf''): resulting l∞ norm must be 1.');
        end

        function testSetnormRms(tc)
            % After setnorm(f,'rms'), rms(f) = norm(f)/sqrt(L) must equal 1.
            f = randn(128, 1);
            fn = setnorm(f, 'rms');
            rms_val = norm(fn) / sqrt(numel(fn));
            tc.verifyEqual(rms_val, 1, 'AbsTol', 1e-12, ...
                'setnorm(''rms''): resulting RMS norm must be 1.');
        end

        function testSetnormPreservesShape(tc)
            % setnorm must not change the size or shape of the input.
            f = randn(32, 4);
            fn = setnorm(f, '2');
            tc.verifyEqual(size(fn), size(f), ...
                'setnorm: output size must equal input size.');
        end

        function testSetnormReturnsNorm(tc)
            % The second output must return the original norm of the input.
            f = randn(64, 1) * 5;
            original_norm = norm(f);
            [~, fnorm] = setnorm(f, '2');
            tc.verifyEqual(fnorm, original_norm, 'AbsTol', 1e-12, ...
                'setnorm: second output must be the original l2 norm of f.');
        end

    end

end
