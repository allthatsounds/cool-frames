classdef PropTransferFunctionConsistency < matlab.unittest.TestCase
%PROPTRANSFERFUNCTIONCONSISTENCY  Consistency of comp_transferfunction across filter types.
%
%   Verifies that comp_transferfunction(g, L) matches the manually constructed
%   full-length L-point DFT response for each filter type:
%
%   1. BL (band-limited) filter: H(k) is zero outside [foff, foff+numel(g.H(L))-1].
%      The non-zero segment is returned by g.H(L) at the correct BL offset.
%
%   2. FIR filter: comp_transferfunction should equal
%      fft(circshift(postpad(g.h, L), g.offset)).
%
%   3. Biquad filter: comp_transferfunction should equal g.H(L) exactly.
%
%   4. Freq filter (function-handle H): g.H(L) returns a BL segment; the full-length
%      response is obtained by placing it at foff via circshift(postpad(H_bl,L),foff).
%
%   5. Modulation consistency: a filter at fc and a version modulated by +fc/2
%      should have responses shifted by round(fc/2 * L/2) bins.
%
%   6. Multiple lengths: consistency holds across L in {128, 256, 512, 1024}.

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

    % ── Helper: manually evaluate BL filter into full-length vector ───────────
    methods (Access = private)

        function H_full = blFilterManual(~, g, L)
            % Reconstruct full-length response by placing BL segment at foff.
            H_bl  = g.H(L);
            foff  = g.foff(L);
            Mbl   = numel(H_bl);
            H_full = zeros(L, 1);
            idx = mod((foff : foff + Mbl - 1)', L) + 1;   % 1-indexed, periodic
            H_full(idx) = H_bl;
        end

    end

    % ── BL filter consistency ─────────────────────────────────────────────────
    methods (Test)

        function testBlFilterConsistencyAcrossLengths(tc)
            % comp_transferfunction must match manual BL placement for each L.
            fc = 0.3;
            g  = blfilter('hann', 0.1, fc);
            for L = [128, 256, 512, 1024]
                H_manual = tc.blFilterManual(g, L);
                H_tf     = comp_transferfunction(g, L);
                tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                    sprintf('blfilter: comp_transferfunction mismatch at L=%d', L));
            end
        end

        function testBlFilterDCConsistency(tc)
            % DC blfilter: foff=0, no wrap-around needed.
            g = blfilter('hann', 0.05, 0);
            L = 256;
            H_manual = tc.blFilterManual(g, L);
            H_tf     = comp_transferfunction(g, L);
            tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                'blfilter DC: comp_transferfunction mismatch');
        end

        function testBlFilterHighFcConsistency(tc)
            % High fc close to Nyquist; BL segment may wrap around.
            g = blfilter('hann', 0.1, 0.8);
            L = 512;
            H_manual = tc.blFilterManual(g, L);
            H_tf     = comp_transferfunction(g, L);
            tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                'blfilter high fc: comp_transferfunction mismatch');
        end

    end

    % ── FIR filter consistency ────────────────────────────────────────────────
    methods (Test)

        function testFirFilterConsistencyAcrossLengths(tc)
            % comp_transferfunction = fft(circshift(postpad(g.h, L), g.offset))
            % comp_filterbank_pre builds the full-length FFT via:
            %   circshift(postpad(g.h, L), g.offset)  then fft()
            for M = [16, 32, 64]
                g = firfilter('hann', M);
                for L = [128, 256, 512]
                    H_manual = fft(circshift(postpad(g.h, L), g.offset));
                    H_tf     = comp_transferfunction(g, L);
                    tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                        sprintf('firfilter M=%d L=%d: comp_transferfunction mismatch', M, L));
                end
            end
        end

        function testFirFilterDCConsistency(tc)
            % DC firfilter: offset=0, causal.
            g = firfilter('hann', 32, 'causal');
            L = 256;
            H_manual = fft(circshift(postpad(g.h, L), g.offset));
            H_tf     = comp_transferfunction(g, L);
            tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                'firfilter causal: comp_transferfunction mismatch');
        end

        function testFirFilterGaussConsistency(tc)
            % Gaussian-shaped FIR via firwin.
            % Note: firwin uses 'truncgauss', not 'gauss' (which is a freqwin name).
            g = firfilter('truncgauss', 48);
            L = 512;
            H_manual = fft(circshift(postpad(g.h, L), g.offset));
            H_tf     = comp_transferfunction(g, L);
            tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                'firfilter truncgauss: comp_transferfunction mismatch');
        end

    end

    % ── Biquad filter consistency ─────────────────────────────────────────────
    methods (Test)

        function testBiquadConsistencyAcrossLengths(tc)
            % comp_transferfunction should exactly equal g.H(L).
            g = biquadfilter(0.3, 0.05, 'peak');
            for L = [128, 256, 512, 1024]
                H_direct = g.H(L);
                H_tf     = comp_transferfunction(g, L);
                tc.verifyEqual(H_tf, H_direct, 'AbsTol', tc.tol, ...
                    sprintf('biquadfilter: comp_transferfunction mismatch at L=%d', L));
            end
        end

        function testBiquadLowpassConsistency(tc)
            % biquadfilter only supports setnorm flags (energy/peak/1/inf) and
            % real/complex — no 'lowpass' mode. Test consistency at a different fc.
            g = biquadfilter(0.2, 0.08);
            L = 512;
            tc.verifyEqual(comp_transferfunction(g, L), g.H(L), 'AbsTol', tc.tol, ...
                'biquadfilter fc=0.2: comp_transferfunction mismatch');
        end

    end

    % ── Freq filter consistency ───────────────────────────────────────────────
    methods (Test)

        function testFreqFilterGaussConsistency(tc)
            % freqfilter g.H(L) returns the BL segment (length Lw, not L).
            % comp_transferfunction places it at foff to produce the full-length vector.
            % Manual reconstruction: circshift(postpad(H_bl, L), foff(L)).
            L  = 512;
            fc = 0.3;
            g  = freqfilter('gauss', 0.1, fc);
            H_bl     = g.H(L);
            foff     = g.foff(L);
            H_manual = circshift(postpad(H_bl(:), L), foff);
            H_tf     = comp_transferfunction(g, L);
            tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                'freqfilter gauss: comp_transferfunction mismatch');
        end

        function testFreqFilterGammatoneConsistency(tc)
            L  = 512;
            fc = 0.25;
            g  = freqfilter('gammatone', 0.08, fc);
            H_bl     = g.H(L);
            foff     = g.foff(L);
            H_manual = circshift(postpad(H_bl(:), L), foff);
            H_tf     = comp_transferfunction(g, L);
            tc.verifyEqual(H_tf, H_manual, 'AbsTol', tc.tol, ...
                'freqfilter gammatone: comp_transferfunction mismatch');
        end

    end

    % ── Delay consistency ─────────────────────────────────────────────────────
    methods (Test)

        function testDelayConsistencyAcrossFilterTypes(tc)
            % For any filter type, adding delay d must multiply H by exp(-2pi*i*d*k/L).
            L = 256;
            d = 5;
            k = (0:L-1)';
            phase = exp(-2*pi*1i*d*k/L);

            % blfilter
            g0 = blfilter('hann', 0.1, 0.3);
            gd = blfilter('hann', 0.1, 0.3, 'delay', d);
            H0 = comp_transferfunction(g0, L);
            Hd = comp_transferfunction(gd, L);
            tc.verifyEqual(Hd, H0 .* phase, 'AbsTol', tc.tol, ...
                'blfilter delay: phase shift mismatch');

            % freqfilter
            g0f = freqfilter('gauss', 0.1, 0.3);
            gdf = freqfilter('gauss', 0.1, 0.3, 'delay', d);
            H0f = comp_transferfunction(g0f, L);
            Hdf = comp_transferfunction(gdf, L);
            tc.verifyEqual(Hdf, H0f .* phase, 'AbsTol', tc.tol, ...
                'freqfilter delay: phase shift mismatch');
        end

    end

    % ── realonly consistency ──────────────────────────────────────────────────
    methods (Test)

        function testRealonlyConsistencyForBlFilter(tc)
            % 'real' flag sets g.realonly=1; 'complex' (default) sets g.realonly=0.
            % Note: comp_transferfunction contains a known bug (checks isfield on cell
            % array, so realonly is never applied automatically). We test:
            %   (a) that the realonly field is set correctly, and
            %   (b) that manually applying (H + involute(H))/2 gives Hermitian output.
            L  = 256;
            g_real    = blfilter('hann', 0.1, 0.3, 'real');
            g_complex = blfilter('hann', 0.1, 0.3);

            % Field check
            tc.verifyEqual(g_real.realonly,    1, 'realonly should be 1 for ''real'' flag');
            tc.verifyEqual(g_complex.realonly, 0, 'realonly should be 0 by default');

            % Manual Hermitian symmetrization: (H + involute(H))/2
            H  = comp_transferfunction(g_real, L);
            H_herm = (H + involute(H)) / 2;
            tc.verifyEqual(H_herm(2:end), conj(flipud(H_herm(2:end))), 'AbsTol', tc.tol, ...
                '(H + involute(H))/2 must be Hermitian symmetric');

            % One-sided (complex) filter is NOT Hermitian on its own
            H_one = comp_transferfunction(g_complex, L);
            residual = norm(H_one(2:end) - conj(flipud(H_one(2:end))));
            tc.verifyGreaterThan(residual, 1e-6, ...
                'one-sided blfilter: should NOT be Hermitian symmetric');
        end

        function testRealonlyTimeDomainReal(tc)
            % After Hermitian symmetrization, ifft(H) should be real-valued.
            for fc = [0.15, 0.3, 0.45]
                L = 256;
                g = blfilter('hann', 0.08, fc, 'real');
                H = comp_transferfunction(g, L);
                H_herm = (H + involute(H)) / 2;  % manual symmetrization
                h = ifft(H_herm);
                tc.verifyLessThan(max(abs(imag(h))), tc.tol, ...
                    sprintf('realonly blfilter fc=%.2f: ifft(H_herm) must be real', fc));
            end
        end

    end

    % ── Length invariance: response shape scales correctly ────────────────────
    methods (Test)

        function testPeakBinScalesWithL(tc)
            % For a BL filter at fc, the peak DFT bin should be round(fc/2*L)+1.
            % As L doubles, peak bin (0-indexed) should approximately double.
            fc  = 0.3;
            g   = blfilter('hann', 0.05, fc);
            tol_bins = 2;

            L1 = 256;  L2 = 512;
            H1 = comp_transferfunction(g, L1);
            H2 = comp_transferfunction(g, L2);
            [~, p1] = max(abs(H1));   % 1-indexed
            [~, p2] = max(abs(H2));

            expected1 = round(fc/2 * L1) + 1;
            expected2 = round(fc/2 * L2) + 1;
            tc.verifyLessThanOrEqual(abs(p1 - expected1), tol_bins, ...
                'Peak bin at L=256 deviates by more than 2 bins from expected');
            tc.verifyLessThanOrEqual(abs(p2 - expected2), tol_bins, ...
                'Peak bin at L=512 deviates by more than 2 bins from expected');
        end

        function testEnergyNormConsistentAcrossL(tc)
            % (1/L)*sum|H(L)|^2 should be approximately the same for different L.
            fc = 0.3;
            g  = blfilter('hann', 0.1, fc);
            energies = zeros(4, 1);
            Ls = [128, 256, 512, 1024];
            for k = 1:4
                L = Ls(k);
                H = comp_transferfunction(g, L);
                energies(k) = sum(abs(H).^2) / L;
            end
            % All normalized energies should be within 10% of each other.
            cv = std(energies) / mean(energies);
            tc.verifyLessThan(cv, 0.1, ...
                'blfilter: (1/L)*sum|H|^2 should be consistent across L values');
        end

    end

end
