classdef TestGabFilters < matlab.unittest.TestCase
%TESTGABFILTERS  Unit tests for gabfilters (linear Gabor filterbank design).
%
%   gabfilters(Ls, g, a, M)
%     -> [gout, aout, fc, L, info]
%
%   Reference:
%     LTFAT layer1/filter_design/gabfilters.m
%
%   Test categories
%   ---------------
%     1. Return-value structure and types
%     2. Filter descriptor fields
%     3. Centre-frequency properties
%     4. Transform length (dgtlength)
%     5. Real vs complex mode
%     6. Time vs freq window axis
%     7. Analysis consistency (ufilterbank equivalence)
%     8. Edge cases and error handling

    properties
        M  = 64
        a  = 16
        Ls = 640
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

        function testReturnsFiveOutputs(tc)
            [gout, aout, fc, L, info] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            tc.verifyNotEmpty(gout, 'gout must not be empty');
            tc.verifyNotEmpty(aout, 'aout must not be empty');
            tc.verifyNotEmpty(fc,   'fc must not be empty');
            tc.verifyGreaterThan(L, 0, 'L must be positive');
            tc.verifyTrue(isstruct(info), 'info must be a struct');
        end

        function testGoutIsCellArray(tc)
            [gout, ~, ~, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            tc.verifyTrue(iscell(gout), 'gout must be a cell array');
        end

        function testInfoHasFcAndTfr(tc)
            [~, ~, ~, ~, info] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            tc.verifyTrue(isfield(info, 'fc'), 'info must have fc field');
            tc.verifyTrue(isfield(info, 'tfr'), 'info must have tfr field');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 2. Filter descriptor fields
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFilterHasRequiredFields(tc)
            [gout, ~, ~, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            for k = 1:numel(gout)
                tc.verifyTrue(isfield(gout{k}, 'H'), 'Filter must have H field');
                tc.verifyTrue(isfield(gout{k}, 'foff'), 'Filter must have foff field');
                tc.verifyTrue(isfield(gout{k}, 'realonly'), 'Filter must have realonly field');
            end
        end

        function testAllFiltersHaveSameHLength(tc)
            [gout, ~, ~, L, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            for k = 1:numel(gout)
                tc.verifyEqual(numel(gout{k}.H), L, ...
                    'All filter H vectors should have length L');
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 3. Centre-frequency properties
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFcLinearlySpaced(tc)
            [~, ~, fc, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            dfc = diff(fc);
            tc.verifyTrue(all(abs(dfc - dfc(1)) < tc.tol), ...
                'Centre frequencies should be linearly spaced');
        end

        function testFcStartsAtZero(tc)
            [~, ~, fc, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            tc.verifyEqual(fc(1), 0, 'AbsTol', tc.tol, ...
                'First centre frequency should be 0');
        end

        function testFcNormalised(tc)
            [~, ~, fc, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            tc.verifyLessThanOrEqual(max(fc), 2, ...
                'All fc values must be in [0, 2)');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 4. Transform length (dgtlength)
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testLIsDgtLength(tc)
            [~, ~, ~, L, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            expectedL = dgtlength(tc.Ls, tc.a, tc.M);
            tc.verifyEqual(L, expectedL, 'L must equal dgtlength(Ls,a,M)');
        end

        function testLDivisibleByAandM(tc)
            [~, ~, ~, L, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            tc.verifyEqual(mod(L, tc.a), 0, 'L must be divisible by a');
            tc.verifyEqual(mod(L, tc.M), 0, 'L must be divisible by M');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 5. Real vs complex mode
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testRealModeFilterCount(tc)
            [gout, ~, fc, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            M2 = floor(tc.M / 2) + 1;
            tc.verifyEqual(numel(gout), M2, ...
                'Real mode should return floor(M/2)+1 filters');
            tc.verifyEqual(numel(fc), M2, ...
                'Real mode fc should have floor(M/2)+1 entries');
        end

        function testComplexModeFilterCount(tc)
            [gout, ~, fc, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M, 'complex');
            tc.verifyEqual(numel(gout), tc.M, ...
                'Complex mode should return M filters');
            tc.verifyEqual(numel(fc), tc.M, ...
                'Complex mode fc should have M entries');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 6. Time vs freq window axis
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testTimeModeProducesValidFilters(tc)
            [gout, ~, ~, L, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M, 'time');
            tc.verifyGreaterThan(sum(abs(gout{1}.H).^2), 0, ...
                'DC filter H must have non-zero energy');
        end

        function testFreqModeProducesValidFilters(tc)
            g0 = randn(tc.M, 1);
            [gout, ~, ~, L, ~] = gabfilters(tc.Ls, g0, tc.a, tc.M, 'freq');
            tc.verifyGreaterThan(sum(abs(gout{1}.H).^2), 0, ...
                'DC filter H must have non-zero energy in freq mode');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 7. Analysis consistency
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFoffSpacing(tc)
            [gout, ~, ~, L, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            foff0 = gout{1}.foff;
            foff1 = gout{2}.foff;
            expectedStep = L / tc.M;
            tc.verifyEqual(foff1 - foff0, expectedStep, 'AbsTol', tc.tol, ...
                'Successive foff values should differ by L/M');
        end

        function testUniformHopSizes(tc)
            [~, aout, ~, ~, ~] = gabfilters(tc.Ls, 'hann', tc.a, tc.M);
            tc.verifyTrue(all(aout(:) == tc.a), ...
                'All hop sizes should equal the specified a');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 8. Edge cases
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testSmallM(tc)
            [gout, ~, fc, L, ~] = gabfilters(64, 'hann', 4, 8);
            M2 = floor(8/2) + 1;
            tc.verifyEqual(numel(gout), M2);
            tc.verifyEqual(numel(fc), M2);
        end

        function testLsEqualsL(tc)
            % When Ls is already a valid DGT length
            L0 = lcm(tc.a, tc.M) * 5;
            [~, ~, ~, L, ~] = gabfilters(L0, 'hann', tc.a, tc.M);
            tc.verifyEqual(L, L0, 'When Ls is valid, L should equal Ls');
        end

        function testNumericWindow(tc)
            g0 = hann(tc.M);
            [gout, ~, ~, ~, ~] = gabfilters(tc.Ls, g0, tc.a, tc.M);
            tc.verifyNotEmpty(gout, 'Must accept numeric window');
        end

    end

end
