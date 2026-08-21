classdef PropGabFilterProperties < matlab.unittest.TestCase
%PROPGABFILTERPROPERTIES  Property-based tests for gabfilters.
%
%   Tests mathematical invariants across random parameter combinations.
%
%   Properties tested
%   -----------------
%     1. Filter count depends only on M and real/complex mode
%     2. Centre frequencies are linearly spaced at 2k/M
%     3. L = dgtlength(Ls, a, M) always holds
%     4. All filters share the same H vector (shifted copies)
%     5. Real/complex mode fc consistency

    properties
        nTrials = 10
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
    % 1. Filter count invariant
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testRealModeFilterCount(tc)
            rng(42);
            for trial = 1:tc.nTrials
                M = 2^randi([3, 8]);
                a = 2^randi([2, 6]);
                Ls = randi([100, 5000]);
                [gout, ~, ~, ~, ~] = gabfilters(Ls, 'hann', a, M);
                M2 = floor(M/2) + 1;
                tc.verifyEqual(numel(gout), M2, ...
                    sprintf('Trial %d: M=%d, expected %d filters', trial, M, M2));
            end
        end

        function testComplexModeFilterCount(tc)
            rng(43);
            for trial = 1:tc.nTrials
                M = 2^randi([3, 8]);
                a = 2^randi([2, 6]);
                Ls = randi([100, 5000]);
                [gout, ~, ~, ~, ~] = gabfilters(Ls, 'hann', a, M, 'complex');
                tc.verifyEqual(numel(gout), M, ...
                    sprintf('Trial %d: M=%d, expected %d filters', trial, M, M));
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 2. Centre frequency linearity
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFcLinearSpacing(tc)
            rng(44);
            for trial = 1:tc.nTrials
                M = 2^randi([3, 8]);
                a = 2^randi([2, 6]);
                Ls = randi([100, 5000]);
                [~, ~, fc, ~, ~] = gabfilters(Ls, 'hann', a, M);
                expected_step = 2/M;
                dfc = diff(fc);
                tc.verifyTrue(all(abs(dfc - expected_step) < tc.tol), ...
                    sprintf('Trial %d: fc spacing should be 2/M=%g', trial, expected_step));
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 3. DGT length invariant
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testDgtLengthInvariant(tc)
            rng(45);
            for trial = 1:tc.nTrials
                M = 2^randi([3, 8]);
                a = 2^randi([2, 6]);
                Ls = randi([100, 5000]);
                [~, ~, ~, L, ~] = gabfilters(Ls, 'hann', a, M);
                expectedL = dgtlength(Ls, a, M);
                tc.verifyEqual(L, expectedL, ...
                    sprintf('Trial %d: L=%d should equal dgtlength=%d', trial, L, expectedL));
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 4. All filters share prototype H
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testAllFiltersSharePrototypeH(tc)
            M = 64; a = 16; Ls = 640;
            [gout, ~, ~, ~, ~] = gabfilters(Ls, 'hann', a, M);
            H0 = gout{1}.H;
            for k = 2:numel(gout)
                tc.verifyEqual(gout{k}.H, H0, ...
                    'All filters should share the same prototype H vector');
            end
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 5. Foff spacing invariant
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testFoffSpacingIsLOverM(tc)
            rng(46);
            for trial = 1:tc.nTrials
                M = 2^randi([3, 8]);
                a = 2^randi([2, 6]);
                Ls = randi([100, 5000]);
                [gout, ~, ~, L, ~] = gabfilters(Ls, 'hann', a, M);
                step = L / M;
                for k = 2:numel(gout)
                    actual_step = gout{k}.foff - gout{k-1}.foff;
                    tc.verifyEqual(actual_step, step, 'AbsTol', tc.tol, ...
                        sprintf('Trial %d: foff step should be L/M=%g', trial, step));
                end
            end
        end

    end

end
