classdef TestFbPhaseGradFromMag < matlab.unittest.TestCase
%TESTFBPHASEGRADFROMMAG  Unit tests for comp_filterbankphasegradfrommag
%   and comp_filterbankneighbors.
%
%   Reference:
%     layer3/phase_processing/comp_filterbankphasegradfrommag.m
%     layer3/phase_processing/comp_filterbankneighbors.m

    properties
        tol = 1e-10
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ─────────────────────────────────────────────────────────────────────────
    % 1. comp_filterbankneighbors — structure
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testNeighShape(tc)
            M = 3; a = [4;8;16]; N = [16;8;4];
            [NEIGH, posInfo] = comp_filterbankneighbors(a, M, N, true);
            Nsum = sum(N);
            tc.verifyEqual(size(NEIGH), [6, Nsum], ...
                'NEIGH must be 6 x Nsum');
            tc.verifyEqual(size(posInfo), [2, Nsum], ...
                'posInfo must be 2 x Nsum');
        end

        function testPosInfoChannelIndices(tc)
            M = 3; a = [4;8;16]; N = [16;8;4];
            [~, posInfo] = comp_filterbankneighbors(a, M, N, true);
            % First N(1) entries should have channel 0 (MATLAB: 0-based after -1)
            chanStart = [0; cumsum(N)];
            for m = 1:M
                idx = chanStart(m)+1:chanStart(m+1);
                tc.verifyTrue(all(posInfo(1,idx) == m-1), ...
                    sprintf('Channel %d posInfo mismatch', m));
            end
        end

        function testPosInfoTimePositions(tc)
            M = 2; a = [4;8]; N = [8;4];
            [~, posInfo] = comp_filterbankneighbors(a, M, N, true);
            % Channel 1: time positions should be 0, 4, 8, ..., 28
            expected1 = (0:N(1)-1)*a(1);
            tc.verifyEqual(posInfo(2, 1:N(1)), expected1, 'AbsTol', tc.tol);
            % Channel 2: time positions should be 0, 8, 16, 24
            expected2 = (0:N(2)-1)*a(2);
            tc.verifyEqual(posInfo(2, N(1)+1:N(1)+N(2)), expected2, 'AbsTol', tc.tol);
        end

        function testNeighHasAboveNeighbors(tc)
            M = 3; a = [4;8;16]; N = [16;8;4];
            [NEIGH, ~] = comp_filterbankneighbors(a, M, N, true);
            % First channel (kk=1) should have above neighbours in rows 5,6
            % (these are 1-based in MATLAB)
            chanStart = [0; cumsum(N)];
            aboveNeigh = NEIGH(5, chanStart(1)+1:chanStart(2));
            tc.verifyTrue(all(aboveNeigh > 0), ...
                'First channel should have above neighbours');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 2. comp_filterbankphasegradfrommag — output shape
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testOutputShapes(tc)
            M = 3; a = [4;8;16]; N = [16;8;4]; Nsum = sum(N);
            fc = [0; 0.5; 1.0]; sqtfr = [0.5; 0.7; 1.0];
            abss = abs(randn(Nsum, 1)) + 0.1;
            [NEIGH, posInfo] = comp_filterbankneighbors(a, M, N, true);
            NEIGH = NEIGH - 1; % MATLAB convention: 0 means no neighbour
            [tgrad, fgrad, logs] = comp_filterbankphasegradfrommag(...
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo, 0.5, 0);
            tc.verifyEqual(size(tgrad), [Nsum, 1]);
            tc.verifyEqual(size(fgrad), [Nsum, 1]);
            tc.verifyEqual(size(logs), [Nsum, 1]);
        end

        function testOutputsAreFinite(tc)
            M = 3; a = [4;8;16]; N = [16;8;4]; Nsum = sum(N);
            fc = [0; 0.5; 1.0]; sqtfr = [0.5; 0.7; 1.0];
            abss = abs(randn(Nsum, 1)) + 0.1;
            [NEIGH, posInfo] = comp_filterbankneighbors(a, M, N, true);
            NEIGH = NEIGH - 1;
            [tgrad, fgrad, logs] = comp_filterbankphasegradfrommag(...
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo, 0.5, 0);
            tc.verifyTrue(all(isfinite(tgrad)), 'tgrad must be finite');
            tc.verifyTrue(all(isfinite(fgrad)), 'fgrad must be finite');
            tc.verifyTrue(all(isfinite(logs)), 'logs must be finite');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 3. logs = log(abss + realmin)
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testLogsIsLogMagnitude(tc)
            M = 2; a = [4;8]; N = [8;4]; Nsum = sum(N);
            fc = [0; 1.0]; sqtfr = [0.5; 1.0];
            abss = abs(randn(Nsum, 1)) + 0.1;
            [NEIGH, posInfo] = comp_filterbankneighbors(a, M, N, true);
            NEIGH = NEIGH - 1;
            [~, ~, logs] = comp_filterbankphasegradfrommag(...
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo, 0.5, 0);
            expected = log(abss + realmin);
            tc.verifyEqual(logs, expected, 'AbsTol', tc.tol);
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 4. Uniform filterbank should approximate comp_ufilterbankphasegradfrommag
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testUniformCase(tc)
            % When all hop sizes are equal, results should be similar
            % (not identical due to different neighbour lookup strategies)
            M = 4; a_val = 8; N_val = 16;
            a = repmat(a_val, M, 1); N = repmat(N_val, M, 1);
            Nsum = sum(N);
            fc = linspace(0, 1, M)';
            sqtfr = ones(M, 1) * 0.8;
            abss = abs(randn(Nsum, 1)) + 0.1;
            [NEIGH, posInfo] = comp_filterbankneighbors(a, M, N, true);
            NEIGH = NEIGH - 1;
            [tgrad, fgrad, ~] = comp_filterbankphasegradfrommag(...
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo, 0.5, 0);
            % Just check it runs and outputs are finite
            tc.verifyTrue(all(isfinite(tgrad)));
            tc.verifyTrue(all(isfinite(fgrad)));
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 5. With tfrdiff
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testTfrdiffChangesResult(tc)
            M = 3; a = [4;8;16]; N = [16;8;4]; Nsum = sum(N);
            fc = [0; 0.5; 1.0]; sqtfr = [0.5; 0.7; 1.0];
            rng(42);
            abss = abs(randn(Nsum, 1)) + 0.1;
            [NEIGH, posInfo] = comp_filterbankneighbors(a, M, N, true);
            NEIGH = NEIGH - 1;
            [tgrad1, ~, ~] = comp_filterbankphasegradfrommag(...
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo, 0.5, 0);
            [tgrad2, ~, ~] = comp_filterbankphasegradfrommag(...
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo, 0.5, 1);
            tc.verifyFalse(isequal(tgrad1, tgrad2), ...
                'do_tfrdiff should change tgrad');
        end

    end

    % ─────────────────────────────────────────────────────────────────────────
    % 6. Constant magnitude should give zero gradients
    % ─────────────────────────────────────────────────────────────────────────
    methods (Test)

        function testConstMagGivesZeroFgrad(tc)
            M = 3; a = [4;8;16]; N = [16;8;4]; Nsum = sum(N);
            fc = [0; 0.5; 1.0]; sqtfr = [0.5; 0.7; 1.0];
            abss = ones(Nsum, 1);
            [NEIGH, posInfo] = comp_filterbankneighbors(a, M, N, true);
            NEIGH = NEIGH - 1;
            [~, fgrad, ~] = comp_filterbankphasegradfrommag(...
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo, 0.5, 0);
            tc.verifyTrue(all(abs(fgrad) < 1e-10), ...
                'Constant magnitude should give zero fgrad');
        end

    end

end
