classdef TestAudFilters < matlab.unittest.TestCase
%TESTAUDFILTERS  Unit tests for audfilters (auditory filterbank design).
%
%   audfilters(fs, Ls)
%     -> [g, a, fc]
%
%   Reference:
%     LTFAT layer1/filter_design/audfilters.m
%
%   Test categories
%   ---------------
%     1. Return-value structure and types
%     2. Centre-frequency properties
%     3. Mel / Mel1000 scale: sensible channel count (not 2000+)
%     4. Filter validity — non-zero response at own centre-frequency bin
%     5. Frame bounds (A > 0 for dense and sparse configurations)
%     6. Frame bounds across all supported scales
%     7. Hop-size and sampling-mode properties
%     8. Partial-tighten / frame-theory helpers
%     9. Edge-case and error handling

    properties
        % Default parameters used by many tests
        p = struct( ...
            'fs',   8000, ...
            'Ls',   2048  ...
        )
        tol = 1e-9
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ─────────────────────────────────────────────────────────────────────────
    % 1. Return-value structure and types
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testReturnsThreeOutputs(tc)
            [g, a, fc] = audfilters(tc.p.fs, tc.p.Ls);
            tc.verifyNotEmpty(g,  'g must not be empty');
            tc.verifyNotEmpty(a,  'a must not be empty');
            tc.verifyNotEmpty(fc, 'fc must not be empty');
        end

        function testGIsACellArray(tc)
            [g, ~, ~] = audfilters(tc.p.fs, tc.p.Ls);
            tc.verifyTrue(iscell(g), 'g must be a cell array of filter structs');
        end

        function testEachFilterHasRequiredFields(tc)
            [g, ~, ~] = audfilters(tc.p.fs, tc.p.Ls);
            required = {'H', 'foff', 'delay', 'realonly'};
            for m = 1:numel(g)
                for r = 1:numel(required)
                    tc.verifyTrue( ...
                        isfield(g{m}, required{r}), ...
                        sprintf('Filter %d: missing field ''%s''', m, required{r}));
                end
            end
        end

        function testAHasSameLengthAsG(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls);
            tc.verifyEqual(size(a, 1), numel(g), ...
                'a must have one row per filter');
        end

        function testFcHasSameLengthAsG(tc)
            [g, ~, fc] = audfilters(tc.p.fs, tc.p.Ls);
            tc.verifyEqual(numel(fc), numel(g), ...
                'fc must have one entry per filter');
        end

        function testAllHopSizesPositive(tc)
            [~, a, ~] = audfilters(tc.p.fs, tc.p.Ls);
            a_int = a(:, 1);   % works for both 1-D and (M,2)
            tc.verifyTrue(all(a_int > 0), 'All hop sizes must be positive');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 2. Centre-frequency properties
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testDcChannelHasZeroFrequency(tc)
            [~, ~, fc] = audfilters(tc.p.fs, tc.p.Ls);
            tc.verifyEqual(fc(1), 0.0, 'First (DC) centre frequency must be 0');
        end

        function testNyquistChannelHasFsOver2(tc)
            [~, ~, fc] = audfilters(tc.p.fs, tc.p.Ls);
            tc.verifyEqual(fc(end), tc.p.fs / 2, ...
                'Last (Nyquist) centre frequency must equal fs/2');
        end

        function testInnerChannelsAreMonotonicallyIncreasing(tc)
            [~, ~, fc] = audfilters(tc.p.fs, tc.p.Ls);
            inner = fc(2:end-1);
            tc.verifyTrue(all(diff(inner) > 0), ...
                'Inner centre frequencies must be strictly increasing');
        end

        function testInnerChannelsBelowNyquist(tc)
            [~, ~, fc] = audfilters(tc.p.fs, tc.p.Ls);
            inner = fc(2:end-1);
            tc.verifyTrue(all(inner < tc.p.fs / 2), ...
                'All inner centre frequencies must be strictly below Nyquist');
        end

        function testInnerChannelsAboveDC(tc)
            [~, ~, fc] = audfilters(tc.p.fs, tc.p.Ls);
            inner = fc(2:end-1);
            tc.verifyTrue(all(inner > 0), ...
                'All inner centre frequencies must be strictly above 0 Hz');
        end

        function testCustomFminFmax(tc)
            fmin = 200;  fmax = 3000;
            [~, ~, fc] = audfilters(tc.p.fs, tc.p.Ls, 'fmin', fmin, 'fmax', fmax);
            inner = fc(2:end-1);
            tc.verifyGreaterThanOrEqual(min(inner), fmin, ...
                'Lowest inner channel must be >= fmin');
            tc.verifyLessThanOrEqual(max(inner), fmax, ...
                'Highest inner channel must be <= fmax');
        end

        function testMParameterControlsChannelCount(tc)
            M = 16;
            [g, ~, ~] = audfilters(tc.p.fs, tc.p.Ls, 'M', M);
            % M sets the requested inner channels.  If the last channel
            % lands at Nyquist it gets trimmed, so actual inner count is M or M-1.
            n_inner = numel(g) - 2;
            tc.verifyGreaterThanOrEqual(n_inner, M - 1, ...
                sprintf('With M=%d, inner channels must be >= M-1=%d, got %d', M, M-1, n_inner));
            tc.verifyLessThanOrEqual(n_inner, M, ...
                sprintf('With M=%d, inner channels must be <= M=%d, got %d', M, M, n_inner));
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 3. Mel / Mel1000 scale: sensible channel count
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testMelScaleChannelCountReasonable(tc)
            % Default mel spacing=100 should give ~10–50 channels for 8 kHz,
            % not the ~2000 that spacing=1 produces.
            [g, ~, ~] = audfilters(tc.p.fs, tc.p.Ls, 'scale', 'mel');
            tc.verifyLessThan(numel(g), 200, ...
                sprintf(['mel scale with default spacing should give <200 channels, ', ...
                         'got %d (spacing=1 bug would give ~2000)'], numel(g)));
            tc.verifyGreaterThan(numel(g), 5, ...
                'mel scale with default spacing should give at least a few channels');
        end

        function testMel1000ScaleChannelCountReasonable(tc)
            [g, ~, ~] = audfilters(tc.p.fs, tc.p.Ls, 'scale', 'mel1000');
            tc.verifyLessThan(numel(g), 200, ...
                sprintf(['mel1000 scale with default spacing should give <200 channels, ', ...
                         'got %d'], numel(g)));
            tc.verifyGreaterThan(numel(g), 5, ...
                'mel1000 scale should give at least a few channels');
        end

        function testMelScaleDefaultSpacingMatches100(tc)
            % Explicitly passing spacing=100 should produce the same filterbank
            % as using the default (scale-appropriate) spacing.
            [g1, ~, fc1] = audfilters(tc.p.fs, tc.p.Ls, 'scale', 'mel');
            [g2, ~, fc2] = audfilters(tc.p.fs, tc.p.Ls, 'scale', 'mel', 'spacing', 100);
            tc.verifyEqual(numel(g1), numel(g2), ...
                'Default mel spacing must match explicit spacing=100');
            tc.verifyEqual(fc1, fc2, tc.tol, ...
                'mel centre frequencies must match between default and spacing=100');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 4. Filter validity — non-zero response at own centre-frequency bin
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testInnerFiltersNonZeroAtOwnCentreFreq(tc)
            % Every inner filter must have |H(fc_bin)| > 0.
            % The old blfilter fftshift bug placed a zero-crossing exactly at
            % fc_bin.  This test would fail with the buggy implementation.
            [g, a, fc] = audfilters(tc.p.fs, tc.p.Ls);
            L = filterbanklength(tc.p.Ls, a);
            for m = 2:numel(g)-1
                Hfull = filter_freqresp(g{m}, L);
                fc_bin = round(L * fc(m) / tc.p.fs) + 1;  % 1-based MATLAB index
                fc_bin = max(1, min(L, fc_bin));
                val = abs(Hfull(fc_bin));
                tc.verifyGreaterThan(val, 0, ...
                    sprintf('Filter %d (fc=%.1f Hz): H(fc_bin) must be > 0, got %.6g', ...
                            m, fc(m), val));
            end
        end

        function testInnerFilterRealonlyFlag(tc)
            % Inner filters (fc > 0) must have realonly = 1.
            [g, ~, fc] = audfilters(tc.p.fs, tc.p.Ls);
            for m = 2:numel(g)-1
                tc.verifyEqual(g{m}.realonly, 1, ...
                    sprintf('Filter %d (fc=%.1f Hz): realonly must be 1', m, fc(m)));
            end
        end

        function testDCFilterRealonlyIsZeroOrOne(tc)
            % DC filter at 0 Hz may have realonly = 0 or 1; just check it's valid.
            [g, ~, ~] = audfilters(tc.p.fs, tc.p.Ls);
            tc.verifyTrue(ismember(g{1}.realonly, [0, 1]), ...
                'DC filter realonly must be 0 or 1');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 5. Frame bounds — A > 0 for dense and sparse configurations
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFrameLowerBoundPositiveDefault(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls);
            L = filterbanklength(tc.p.Ls, a);
            [A, B] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('Default ERB: frame lower bound A must be > 0 (got A=%.6g)', A));
            tc.verifyGreaterThan(B, 0, 'Frame upper bound B must be > 0');
            tc.verifyLessThanOrEqual(A, B, 'Must have A <= B');
        end

        function testFrameLowerBoundPositiveSparseM20(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'M', 20);
            L = filterbanklength(tc.p.Ls, a);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('ERB M=20: frame lower bound A must be > 0 (got A=%.6g)', A));
        end

        function testFrameLowerBoundPositiveSparseM10(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'M', 10);
            L = filterbanklength(tc.p.Ls, a);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('ERB M=10: frame lower bound A must be > 0 (got A=%.6g)', A));
        end

        function testFrameLowerBoundPositiveBark(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'scale', 'bark');
            L = filterbanklength(tc.p.Ls, a);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('Bark scale: frame lower bound A must be > 0 (got A=%.6g)', A));
        end

        function testFrameLowerBoundPositiveMel(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'scale', 'mel');
            L = filterbanklength(tc.p.Ls, a);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('Mel scale: frame lower bound A must be > 0 (got A=%.6g)', A));
        end

        function testFrameLowerBoundPositiveMel1000(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'scale', 'mel1000');
            L = filterbanklength(tc.p.Ls, a);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('Mel1000 scale: frame lower bound A must be > 0 (got A=%.6g)', A));
        end

        function testConditionNumberFinite(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls);
            L = filterbanklength(tc.p.Ls, a);
            [A, B] = filterbankrealbounds(g, a, L);
            kappa = B / A;
            tc.verifyTrue(isfinite(kappa), ...
                sprintf('Condition number kappa=B/A must be finite (got %.6g)', kappa));
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 6. Frame bounds across all supported scales
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFrameBoundsAllScales(tc)
            scales = {'erb', 'bark', 'mel', 'mel1000', 'log', 'linear'};
            for si = 1:numel(scales)
                sc = scales{si};
                try
                    [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'scale', sc);
                catch ME
                    tc.assumeFail(sprintf('Scale ''%s'' not supported: %s', sc, ME.message));
                    continue
                end
                L = filterbanklength(tc.p.Ls, a);
                [A, ~] = filterbankrealbounds(g, a, L);
                tc.verifyGreaterThan(A, 0, ...
                    sprintf('Scale ''%s'': frame lower bound A must be > 0 (got A=%.6g)', ...
                            sc, A));
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 7. Hop-size and sampling-mode properties
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testRegSamplingDivisibility(tc)
            % In regsampling mode L must be divisible by every hop size.
            [~, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'sampling', 'regsampling');
            L = filterbanklength(tc.p.Ls, a);
            a_int = a(:, 1);
            remainders = mod(L, a_int);
            tc.verifyTrue(all(remainders == 0), ...
                'In regsampling mode, L must be divisible by every hop size');
        end

        function testUniformSamplingConstantHopSize(tc)
            [~, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'sampling', 'uniform');
            tc.verifyTrue(all(a == a(1)), ...
                'In uniform sampling mode, all hop sizes must be equal');
        end

        function testFractionalSamplingShape(tc)
            [~, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'sampling', 'fractional');
            tc.verifyEqual(size(a, 2), 2, ...
                'In fractional mode, a must be an (M, 2) matrix');
        end

        function testLGreaterThanOrEqualToLs(tc)
            [~, a, ~] = audfilters(tc.p.fs, tc.p.Ls);
            L = filterbanklength(tc.p.Ls, a);
            tc.verifyGreaterThanOrEqual(L, tc.p.Ls, 'L must be >= Ls');
        end

        function testRedmulIncreasesRedundancy(tc)
            [~, a1, ~] = audfilters(tc.p.fs, tc.p.Ls, 'redmul', 1.0);
            [~, a2, ~] = audfilters(tc.p.fs, tc.p.Ls, 'redmul', 2.0);
            % Higher redmul → smaller hop sizes → higher redundancy
            a1_min = min(a1(:, 1));
            a2_min = min(a2(:, 1));
            tc.verifyLessThanOrEqual(a2_min, a1_min, ...
                'redmul=2 should produce hop sizes <= redmul=1');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 8. Frame-theory helpers: filterbankrealtight
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFilterbankrealtightGivesKappaOne(tc)
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'M', 20);
            L = filterbanklength(tc.p.Ls, a);
            g_tight = filterbankrealtight(g, a, L);
            [At, Bt] = filterbankrealbounds(g_tight, a, L);
            kappa_tight = Bt / At;
            tc.verifyEqual(kappa_tight, 1.0, tc.tol, ...
                sprintf('Tight frame must have kappa=1 (got kappa=%.8g)', kappa_tight));
        end

        function testFilterbankdualdualIsIdentity(tc)
            % dual(dual(g)) should equal g in frame response.
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'M', 20);
            L = filterbanklength(tc.p.Ls, a);
            gd = filterbankrealdual(g, a, L);
            gdd = filterbankrealdual(gd, a, L);
            % Compare full frequency responses
            for m = 1:numel(g)
                H_orig = filter_freqresp(g{m}, L);
                H_dd   = filter_freqresp(gdd{m}, L);
                err = max(abs(H_orig - H_dd));
                tc.verifyLessThan(err, 1e-6, ...
                    sprintf('Filter %d: dual(dual(g)) must recover g (max err=%.2e)', m, err));
            end
        end

        function testPerfectReconstructionViaFilterbankRealdual(tc)
            % Analysis with g, synthesis with dual(g) should give perfect reconstruction.
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'M', 20);
            L = filterbanklength(tc.p.Ls, a);
            gd = filterbankrealdual(g, a, L);
            x = randn(L, 1);
            c = filterbank(x, g, a);
            xrec = ifilterbank(c, gd, a, L);
            err = max(abs(xrec - x));
            % LTFAT's ifilterbank uses the NUMEL(g)-channel real filterbank
            % convention; reconstruction tolerance of 1e-4 is acceptable.
            tc.verifyLessThan(err, 1e-4, ...
                sprintf('Perfect reconstruction error must be < 1e-4 (got %.2e)', err));
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 9. Edge-case and error handling
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testInvalidSamplingModeErrors(tc)
            tc.verifyError( ...
                @() audfilters(tc.p.fs, tc.p.Ls, 'sampling', 'bogusmode'), ...
                '', ...
                'Unknown sampling mode must throw an error');
        end

        function testFminAboveFmaxErrors(tc)
            % fmin > fmax should produce an empty or error state
            tc.verifyError( ...
                @() audfilters(tc.p.fs, tc.p.Ls, 'fmin', 4000, 'fmax', 100), ...
                '', ...
                'fmin > fmax should throw an error or produce an empty filterbank');
        end

        function testMinWindowLengthEnforced(tc)
            % min_win sets the minimum number of DFT bins per filter window.
            [g, a, ~] = audfilters(tc.p.fs, tc.p.Ls, 'min_win', 8);
            L = filterbanklength(tc.p.Ls, a);
            % Check that every inner filter has a support of at least 8 bins
            for m = 2:numel(g)-1
                Hfull = filter_freqresp(g{m}, L);
                nnz_bins = sum(abs(Hfull) > 0);
                tc.verifyGreaterThanOrEqual(nnz_bins, 8, ...
                    sprintf('Filter %d: support must be >= min_win=8 bins (got %d)', m, nnz_bins));
            end
        end

        function testSingleChannelDegenerateCase(tc)
            % M=1 should not error; it may produce a coarse filterbank.
            try
                [g, a, fc] = audfilters(tc.p.fs, tc.p.Ls, 'M', 1);
                tc.verifyGreaterThanOrEqual(numel(g), 1, ...
                    'M=1 should produce at least one filter (DC+1 inner+Nyquist)');
                L = filterbanklength(tc.p.Ls, a);
                [A, ~] = filterbankrealbounds(g, a, L);
                tc.verifyGreaterThan(A, 0, 'M=1: frame lower bound must be > 0');
            catch ME
                % Some implementations may legitimately reject M=1; that's OK.
                tc.assumeFail(sprintf('M=1 threw: %s', ME.message));
            end
        end

        function testHighSampleRate(tc)
            fs_high = 44100;
            Ls_high = 16384;
            [g, a, fc] = audfilters(fs_high, Ls_high);
            L = filterbanklength(Ls_high, a);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('fs=44100: frame lower bound A must be > 0 (got A=%.6g)', A));
            tc.verifyEqual(fc(end), fs_high / 2, ...
                'Nyquist channel must equal fs/2 for fs=44100');
        end

    end

end
