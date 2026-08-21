classdef TestCqtFilters < matlab.unittest.TestCase
%TESTCQTFILTERS  Unit tests for cqtfilters (constant-Q filterbank design).
%
%   cqtfilters(fs, fmin, fmax, bins, Ls)
%     -> [g, a, fc, L, info]
%
%   Reference:
%     LTFAT layer1/filter_design/cqtfilters.m
%     (Holighaus, Velasco; modified by Prusa, 2013–2014)
%
%   Test categories
%   ---------------
%     1. Return-value structure and types
%     2. Centre-frequency properties
%     3. Bandwidth / constant-Q property
%     4. Filter validity (non-zero response, realonly flag)
%     5. Frame bounds (A > 0 so the system is a frame)
%     6. Hop-size properties
%     7. Optional-parameter effects (Qvar, redmul, sampling modes)
%     8. Edge-case / error handling

    properties
        % Default parameters used by many tests
        p = struct( ...
            'fs',   8000, ...
            'fmin',  100, ...
            'fmax', 3500, ...
            'bins',   12, ...
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

        function testReturnsFiveCellOutputs(tc)
            [g, a, fc, L, info] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyNotEmpty(g,    'g must not be empty');
            tc.verifyNotEmpty(a,    'a must not be empty');
            tc.verifyNotEmpty(fc,   'fc must not be empty');
            tc.verifyNotEmpty(L,    'L must be a scalar');
            tc.verifyNotEmpty(info, 'info must be non-empty');
        end

        function testGIsACellArray(tc)
            [g, ~, ~, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyTrue(iscell(g), 'g must be a cell array of filter structs');
        end

        function testEachFilterHasRequiredFields(tc)
            [g, ~, ~, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            required = {'H', 'foff', 'delay', 'realonly'};
            for m = 1:numel(g)
                for r = 1:numel(required)
                    tc.verifyTrue( ...
                        isfield(g{m}, required{r}), ...
                        sprintf('Filter %d: missing field ''%s''', m, required{r}));
                end
            end
        end

        function testLIsScalarPositiveInteger(tc)
            [~, ~, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyTrue(isscalar(L),  'L must be scalar');
            tc.verifyTrue(L > 0,        'L must be positive');
            tc.verifyTrue(L >= tc.p.Ls, 'L must be >= Ls');
            tc.verifyEqual(mod(L, 1), 0, 'L must be an integer');
        end

        function testFcAndGAndAHaveSameLength(tc)
            [g, a, fc, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            M = numel(g);
            tc.verifyEqual(numel(fc), M, 'numel(fc) must equal numel(g)');
            tc.verifyEqual(size(a, 1), M, 'size(a,1) must equal numel(g)');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 2. Centre-frequency properties
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFirstCentreFreqIsDC(tc)
            [~, ~, fc, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyEqual(fc(1), 0, 'First centre frequency must be 0 Hz (DC)');
        end

        function testLastCentreFreqIsNyquist(tc)
            [~, ~, fc, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyEqual(fc(end), tc.p.fs / 2, ...
                'Last centre frequency must be Nyquist = fs/2');
        end

        function testCentreFreqsAreMonotoneIncreasing(tc)
            [~, ~, fc, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyTrue(all(diff(fc) > 0), ...
                'Centre frequencies must be strictly monotone increasing');
        end

        function testInnerFreqsStartNearFmin(tc)
            [~, ~, fc, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            % fc(2) should be very close to fmin
            tc.verifyLessThanOrEqual( ...
                abs(fc(2) - tc.p.fmin) / tc.p.fmin, 0.01, ...
                'Second channel (first CQT) must be near fmin');
        end

        function testInnerFreqsDoNotExceedNyquist(tc)
            [~, ~, fc, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyTrue(all(fc <= tc.p.fs / 2 + tc.tol), ...
                'No centre frequency may exceed Nyquist');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 3. Bandwidth / constant-Q property
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testConstantQOverInnerChannels(tc)
            %TESTCONSTANTQOVERINNERCHANNELS  Q = fc/bw should be constant.
            [~, ~, fc, ~, info] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            % Inner channels: skip DC (1) and Nyquist (end)
            fc_inner = fc(2:end-1);
            if isfield(info, 'fsupp')
                bw_inner = info.fsupp(2:end-1);
                Q_vals = fc_inner ./ bw_inner;
                Q_cv = std(Q_vals) / mean(Q_vals);  % coefficient of variation
                tc.verifyLessThan(Q_cv, 0.05, ...
                    sprintf('Q coefficient of variation too large: %.4f', Q_cv));
            else
                % Approximate Q from adjacent centre frequencies
                if numel(fc_inner) >= 3
                    bw_approx = fc_inner(3:end) - fc_inner(1:end-2);
                    Q_approx  = fc_inner(2:end-1) ./ bw_approx;
                    Q_cv = std(Q_approx) / mean(Q_approx);
                    tc.verifyLessThan(Q_cv, 0.05, ...
                        sprintf('Q (approximate) CV too large: %.4f', Q_cv));
                end
            end
        end

        function testExpectedQForBins12(tc)
            %TESTEXPECTEDQFORBINS12  Q_natural = 1 / (2^(1/12) - 2^(-1/12)) ~ 8.67
            [~, ~, fc, ~, info] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, 12, tc.p.Ls);
            bins = 12;
            Q_nat = 1 / (2^(1/bins) - 2^(-1/bins));
            if isfield(info, 'fsupp')
                bw_inner = info.fsupp(2:end-1);
                fc_inner = fc(2:end-1);
                Q_vals = fc_inner ./ bw_inner;
                tc.verifyTrue( ...
                    all(abs(Q_vals - Q_nat) / Q_nat < 0.05), ...
                    sprintf('Q values far from Q_natural=%.2f', Q_nat));
            end
        end

        function testQvarScalesBandwidth(tc)
            Qvar1 = 1.0;
            Qvar2 = 2.0;
            [~, ~, ~, ~, info1] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls, 'Qvar', Qvar1);
            [~, ~, ~, ~, info2] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls, 'Qvar', Qvar2);
            if isfield(info1, 'fsupp') && isfield(info2, 'fsupp')
                % Inner bandwidths should be scaled by Qvar2 / Qvar1
                bw1 = info1.fsupp(2:end-1);
                bw2 = info2.fsupp(2:end-1);
                ratio = bw2 ./ bw1;
                tc.verifyTrue( ...
                    all(abs(ratio - Qvar2/Qvar1) < 0.01), ...
                    'Qvar should linearly scale inner bandwidths');
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 4. Filter validity
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testDCFilterHasRealonly0(tc)
            [g, ~, ~, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            tc.verifyEqual(g{1}.realonly, 0, ...
                'DC filter (fc=0) must have realonly=0');
        end

        function testInnerFiltersHaveRealonly1(tc)
            [g, ~, ~, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            for m = 2:numel(g)-1
                tc.verifyEqual(g{m}.realonly, 1, ...
                    sprintf('Inner filter %d must have realonly=1', m));
            end
        end

        function testAllFiltersHaveFiniteTransferFunction(tc)
            [g, ~, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            for m = 1:numel(g)
                H = comp_transferfunction(g{m}, L);
                tc.verifyTrue(all(isfinite(H)), ...
                    sprintf('Filter %d: non-finite values in transfer function', m));
            end
        end

        function testNoFilterIsAllZero(tc)
            [g, ~, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            for m = 1:numel(g)
                H = comp_transferfunction(g{m}, L);
                tc.verifyGreaterThan(max(abs(H)), 0, ...
                    sprintf('Filter %d is all-zero', m));
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 5. Frame bounds (key correctness property)
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFrameLowerBoundPositive(tc)
            %TESTFRAMELOWERBOUNDPOSITIVE  A > 0 means system is a frame.
            [g, a, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            [A, B] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                sprintf('Frame lower bound A must be > 0 (got A=%.6g)', A));
            tc.verifyGreaterThan(B, A, 'Frame upper bound B must exceed A');
        end

        function testFrameBoundsFinite(tc)
            [g, a, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            [A, B] = filterbankrealbounds(g, a, L);
            tc.verifyTrue(isfinite(A), 'Frame lower bound A must be finite');
            tc.verifyTrue(isfinite(B), 'Frame upper bound B must be finite');
        end

        function testFrameBoundsWithQvar2(tc)
            %TESTFRAMEBOUNDSWITHQVAR2  Wider bands → better frame condition.
            [g, a, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls, 'Qvar', 2.0);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                'Frame must still be valid for Qvar=2.0');
        end

        function testFrameBoundsWithRedmul2(tc)
            [g, a, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls, 'redmul', 2.0);
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                'Frame must still be valid for redmul=2.0');
        end

        function testTightFilterbankFromFilterbanktight(tc)
            %TESTTIGHTFILTERBANKFROMFILTERBANKTIGHT  After tightening, A==B.
            [g, a, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            gt = filterbanktight(g, a, L);
            [At, Bt] = filterbankrealbounds(gt, a, L);
            tc.verifyLessThan( ...
                abs(At - Bt) / max(abs(At), 1e-12), 1e-6, ...
                sprintf('Tight filterbank: A=%g, B=%g should be equal', At, Bt));
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 6. Hop-size properties
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testAllHopSizesPositive(tc)
            [~, a, ~, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            a_vals = a(:, 1);    % numerator column (or full 1-D array)
            tc.verifyTrue(all(a_vals > 0), 'All hop sizes must be positive');
        end

        function testLIsDivisibleByAllHopSizes(tc)
            [~, a, ~, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls);
            if isvector(a) && ~ismatrix(a(:, :))
                a_num = a;
            else
                a_num = a(:, 1);
            end
            for m = 1:numel(a_num)
                tc.verifyEqual(mod(L, a_num(m)), 0, ...
                    sprintf('L=%d is not divisible by a(%d)=%d', L, m, a_num(m)));
            end
        end

        function testUniformSamplingModeProducesUniformHops(tc)
            [~, a, ~, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls, ...
                'sampling', 'uniform');
            if isvector(a)
                tc.verifyEqual(numel(unique(a)), 1, ...
                    'Uniform sampling: all hop sizes should be equal');
            end
        end

        function testFractionalSamplingModeReturnsTwoColumns(tc)
            [~, a, ~, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, tc.p.bins, tc.p.Ls, ...
                'sampling', 'fractional');
            tc.verifyEqual(size(a, 2), 2, ...
                'Fractional sampling: a must be an (M+2, 2) array');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 7. Optional-parameter effects
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testVariableBinsPerOctave(tc)
            %TESTVARIABLEBINSPEROCTAVE  bins can be a vector (one per octave).
            bins_vec = [8, 12, 16];  % per-octave bins
            [g, a, fc, L, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fmax, bins_vec, tc.p.Ls);
            tc.verifyNotEmpty(g, 'g must not be empty for vector bins');
            [A, ~] = filterbankrealbounds(g, a, L);
            tc.verifyGreaterThan(A, 0, ...
                'Frame must be valid for variable bins per octave');
        end

        function testDifferentFsValues(tc)
            for fs = [4000, 16000, 44100]
                [g, a, fc, L, ~] = cqtfilters( ...
                    fs, tc.p.fmin, min(tc.p.fmax, fs/2-1), tc.p.bins, tc.p.Ls);
                [A, ~] = filterbankrealbounds(g, a, L);
                tc.verifyGreaterThan(A, 0, ...
                    sprintf('Frame invalid for fs=%d', fs));
            end
        end

        function testFmaxClippedToNyquist(tc)
            %TESTFMAXCLIPPEDTONYQUIST  fmax > fs/2 should not raise an error.
            [~, ~, fc, ~, ~] = cqtfilters( ...
                tc.p.fs, tc.p.fmin, tc.p.fs,  % fmax = fs (> Nyquist)
                tc.p.bins, tc.p.Ls);
            tc.verifyLessThanOrEqual(max(fc), tc.p.fs / 2 + tc.tol, ...
                'Centre frequencies must be clipped to Nyquist');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 8. Error handling
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testErrorWhenFminGeqFmax(tc)
            tc.verifyError( ...
                @() cqtfilters(tc.p.fs, 1000, 500, tc.p.bins, tc.p.Ls), ...
                ?MException, ...
                'Should raise an error when fmin >= fmax');
        end

        function testErrorWhenFminIsZero(tc)
            tc.verifyError( ...
                @() cqtfilters(tc.p.fs, 0, tc.p.fmax, tc.p.bins, tc.p.Ls), ...
                ?MException, ...
                'Should raise an error when fmin = 0');
        end

    end

end
