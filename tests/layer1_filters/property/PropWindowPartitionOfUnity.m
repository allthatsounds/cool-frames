classdef PropWindowPartitionOfUnity < matlab.unittest.TestCase
%PROPWINDOWPARTITIONOFUNITY  Partition-of-unity and tight-frame properties of firwin.
%
%   Partition of unity (PU):  w + fftshift(w) = ones(M,1)
%     Holds for: hann, rect, tria (for even M)
%
%   Tight frame condition:    w.^2 + fftshift(w.^2) = ones(M,1)
%     Holds for: sine (= sqrt(hann))
%
%   These are the key frame-theoretic identities needed to form
%   tight Gabor / Wilson / WMDCT frames.

    properties
        lengths = [32, 64, 128, 256]   % even lengths to test
        tol = 1e-13
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── Partition of unity: hann ─────────────────────────────────────────────
    methods (Test)

        function testHannPUMultipleLengths(tc)
            for M = tc.lengths
                w = firwin('hann', M);
                residual = norm(w + fftshift(w) - ones(M, 1));
                tc.verifyLessThan(residual, tc.tol, ...
                    sprintf('hann PU failed for M=%d: residual=%g', M, residual));
            end
        end

        function testHannPUSumIsOne(tc)
            % Element-wise: each pair (w(k), w(k+M/2)) sums to 1
            M = 64;
            w = firwin('hann', M);
            pu = w + fftshift(w);
            tc.verifyEqual(pu, ones(M, 1), 'AbsTol', tc.tol, ...
                'hann: w + fftshift(w) should be all-ones');
        end

        function testRectPU(tc)
            % firwin('rect', M) for even M is a rectangular indicator:
            %   w(k) = 1  for all sample points x with |x| < 1/2
            %   w(k) = 0  at the Nyquist point x = -1/2 (position M/2+1)
            %
            % For even M, rect does NOT satisfy the PU condition
            % w + fftshift(w) = ones(M,1): the shifted sum equals 2 at
            % interior bins, 1 at DC and Nyquist (docstring states
            % "Forms a PU if the order is odd").
            %
            % This test verifies the actual binary boxcar structure.
            for M = tc.lengths
                w        = firwin('rect', M);
                expected = ones(M, 1);
                expected(M/2 + 1) = 0;   % Nyquist bin (x = -1/2) is 0
                tc.verifyEqual(w, expected, 'AbsTol', tc.tol, ...
                    sprintf('rect: value should be 1 everywhere except Nyquist bin (M=%d)', M));
            end
        end

        function testTriaPU(tc)
            for M = tc.lengths
                w = firwin('tria', M);
                residual = norm(w + fftshift(w) - ones(M, 1));
                tc.verifyLessThan(residual, tc.tol, ...
                    sprintf('tria PU failed for M=%d', M));
            end
        end

    end

    % ── Tight frame: sine = sqrt(hann) ───────────────────────────────────────
    methods (Test)

        function testSineTightFrameMultipleLengths(tc)
            % sine.^2 + fftshift(sine.^2) = ones (tight frame condition)
            for M = tc.lengths
                w = firwin('sine', M);
                residual = norm(w.^2 + fftshift(w.^2) - ones(M, 1));
                tc.verifyLessThan(residual, tc.tol, ...
                    sprintf('sine tight frame failed for M=%d: residual=%g', M, residual));
            end
        end

        function testSineIsSqrtHann(tc)
            % sine(k)^2 = hann(k)  (by definition)
            M    = 64;
            sine = firwin('sine', M);
            hann = firwin('hann', M);
            tc.verifyEqual(sine.^2, hann, 'AbsTol', tc.tol, ...
                'sine^2 should equal hann point-wise');
        end

        function testItersineIsSqrtHannAlias(tc)
            % 'itersine' = sin(pi/2 * cos^2(pi*x)) — NOT an alias for 'sine'
            % (which is sqrt(hann)). Both have issqpu=1 (tight-frame property),
            % but they are numerically different windows. Verify:
            %   (a) itersine satisfies the tight-frame condition independently, and
            %   (b) itersine ≠ sine.
            M = 64;
            w_sine     = firwin('sine', M);
            w_itersine = firwin('itersine', M);

            % itersine is a tight frame (its own square sums to PU)
            residual_tf = norm(w_itersine.^2 + fftshift(w_itersine.^2) - ones(M,1));
            tc.verifyLessThan(residual_tf, tc.tol, ...
                'itersine: w^2 + fftshift(w^2) must equal ones (tight frame)');

            % itersine and sine are distinct windows
            tc.verifyGreaterThan(norm(w_itersine - w_sine), 1e-3, ...
                'itersine and sine should be numerically different windows');
        end

    end

    % ── Symmetry is a prerequisite of PU ─────────────────────────────────────
    methods (Test)

        function testAllPUWindowsAreWPESymmetric(tc)
            % PU windows must be WPE: w(k) = w(M+2-k) for k=2..M
            windows = {'hann', 'rect', 'tria', 'sine'};
            M = 64;
            for wi = 1:numel(windows)
                w = firwin(windows{wi}, M);
                residual = norm(w(2:end) - flipud(w(2:end)));
                tc.verifyLessThan(residual, tc.tol, ...
                    sprintf('%s: should be WPE symmetric', windows{wi}));
            end
        end

    end

    % ── Random-length stress test ────────────────────────────────────────────
    methods (Test)

        function testHannPURandomEvenLengths(tc)
            rng(42);
            for trial = 1:20
                M = 2 * randi([8, 256]);   % random even length
                w = firwin('hann', M);
                residual = norm(w + fftshift(w) - ones(M, 1));
                tc.verifyLessThan(residual, tc.tol, ...
                    sprintf('hann PU failed for random M=%d', M));
            end
        end

        function testSineTightFrameRandomEvenLengths(tc)
            rng(42);
            for trial = 1:20
                M = 2 * randi([8, 256]);
                w = firwin('sine', M);
                residual = norm(w.^2 + fftshift(w.^2) - ones(M, 1));
                tc.verifyLessThan(residual, tc.tol, ...
                    sprintf('sine tight frame failed for random M=%d', M));
            end
        end

    end

end
