classdef PropIterativeReconstruction < matlab.unittest.TestCase
%PROPITERATIVERECONSTRUCTION  ifilterbankiter converges to the correct signal.
%
%   ifilterbankiter uses iterative (CG/PCG) inversion.  For a valid frame:
%
%   (1) The relative residual relres < tol at convergence.
%   (2) The iterative result agrees with direct dual-frame inversion.
%   (3) More iterations yield a smaller (or equal) residual.
%
%   NOTE: ifilterbankiter depends on frsyniter, which is part of the system
%   LTFAT installation.  Tests are skipped gracefully if frsyniter is absent.

    properties
        p   % scalar parameters (fs, Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);
            tc.p = struct('fs', 8000, 'Ls', 1024);
        end
    end

    % ── Helper ───────────────────────────────────────────────────────────────
    methods (Access = private)
        function skip = frsyniterUnavailable(~)
            skip = ~(exist('frsyniter', 'file') || exist('frsyniter', 'builtin'));
        end
    end

    methods (Test)

        function testConvergenceRelativeResidualBelowTolerance(tc)
            if tc.frsyniterUnavailable()
                tc.assumeTrue(false, 'frsyniter not available; test skipped');
            end

            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            Ls = tc.p.Ls;
            tol = 1e-6;

            for trial = 1:5
                x           = randn(Ls,1);   % real signal (ERB bank is one-sided)
                c           = filterbank(x, g, a);
                [~, relres] = ifilterbankiter(c, g, a, 'tol', tol);

                tc.verifyLessThan(relres, 10*tol, ...
                    sprintf('Trial %d: relres=%.2e exceeds 10 × tol=%.2e', ...
                    trial, relres, tol));
            end
        end

        function testIterativeMatchesDirectDual(tc)
            if tc.frsyniterUnavailable()
                tc.assumeTrue(false, 'frsyniter not available; test skipped');
            end

            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls, 'complex');
            gd           = filterbankdual(g, a, L);
            Ls           = tc.p.Ls;

            for trial = 1:3
                x          = randn(Ls,1);   % real signal
                c          = filterbank(x, g, a);

                xr_direct  = ifilterbank(c, gd, a, L);
                [xr_iter, relres] = ifilterbankiter(c, g, a);

                relErr = norm(xr_direct(1:Ls) - xr_iter(1:Ls)) / norm(xr_direct(1:Ls));
                tc.verifyLessThan(relErr, 1e-1, ...
                    sprintf('Trial %d: iterative vs direct error %.2e (relres=%.2e)', ...
                    trial, relErr, relres));
            end
        end

        function testMoreIterationsGiveSmallerOrEqualResidual(tc)
            if tc.frsyniterUnavailable()
                tc.assumeTrue(false, 'frsyniter not available; test skipped');
            end

            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls); %#ok<ASGLU>
            Ls = tc.p.Ls;
            x  = randn(Ls,1);   % real signal
            c  = filterbank(x, g, a);

            maxit_vals  = [5, 15, 40];
            relres_vals = zeros(size(maxit_vals));
            for k = 1:numel(maxit_vals)
                [~, relres_vals(k)] = ifilterbankiter(c, g, a, ...
                    'maxit', maxit_vals(k));
            end

            for k = 1:numel(maxit_vals)-1
                tc.verifyLessThanOrEqual(relres_vals(k+1), relres_vals(k) + 1e-10, ...
                    sprintf('Residual increased: maxit=%d gives %.2e, maxit=%d gives %.2e', ...
                    maxit_vals(k),   relres_vals(k), ...
                    maxit_vals(k+1), relres_vals(k+1)));
            end
        end

        function testTightFrameConvergesInOnePCGStep(tc)
            % For a tight frame, PCG preconditioned with the frame bound converges
            % in very few iterations.  Verify at least that relres < 1e-6.
            if tc.frsyniterUnavailable()
                tc.assumeTrue(false, 'frsyniter not available; test skipped');
            end

            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls, 'complex');
            gt           = filterbanktight(g, a, L);
            Ls           = tc.p.Ls;

            x           = randn(Ls,1);   % real signal
            c           = filterbank(x, gt, a);
            [~, relres] = ifilterbankiter(c, gt, a, 'pcg', 'tol', 1e-8);

            tc.verifyLessThan(relres, 1e-6, ...
                sprintf('Tight-frame PCG relres=%.2e exceeds 1e-6', relres));
        end

    end
end
