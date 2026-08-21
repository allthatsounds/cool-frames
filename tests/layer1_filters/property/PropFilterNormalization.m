classdef PropFilterNormalization < matlab.unittest.TestCase
%PROPFILTERNORMALIZATION  Normalization consistency across all filter constructors.
%
%   For every constructor (blfilter, firfilter, freqfilter, biquadfilter):
%
%   'energy' norm: (1/L) * sum|H(L)|^2 = 1  for any valid L
%   'peak'   norm: max|H(L)| = 1             for any valid L
%   'scal'   s:    H_scal = s * H_default    (linear in s)
%
%   These should hold across different signal lengths L.

    properties
        tol_energy = 1e-8
        tol_peak   = 1e-10
        lengths    = [128, 256, 512, 1024]
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── Energy normalization across lengths ───────────────────────────────────
    methods (Test)

        function testBlFilterEnergyNormMultipleLengths(tc)
            g = blfilter('hann', 0.1, 0.3, 'energy');
            for L = tc.lengths
                H      = comp_transferfunction(g, L);
                energy = sum(abs(H).^2) / L;
                tc.verifyEqual(energy, 1, 'AbsTol', tc.tol_energy, ...
                    sprintf('blfilter energy norm failed at L=%d', L));
            end
        end

        function testFreqFilterEnergyNormMultipleLengths(tc)
            % freqfilter 'energy' passes 'energy' to freqwin, which
            % normalises the BL segment to unit L2 norm (sum|h|^2=1).
            % After multiplying by sqrt(L) the full-length response
            % satisfies (1/L)*sum|H|^2 = 1, matching blfilter behaviour.
            g = freqfilter('gauss', 0.05, 0.3, 'energy');
            for L = tc.lengths
                H = comp_transferfunction(g, L);
                energy = sum(abs(H).^2) / L;
                tc.verifyEqual(energy, 1, 'AbsTol', tc.tol_energy, ...
                    sprintf('freqfilter energy norm failed at L=%d', L));
            end
        end

        function testFirFilterEnergyNormMultipleLengths(tc)
            % firfilter energy: sum(h.^2) = 1 (length-independent in time domain)
            for M = [16, 32, 64]
                g = firfilter('hann', M, 0, 'energy');
                tc.verifyEqual(sum(g.h.^2), 1, 'AbsTol', tc.tol_energy, ...
                    sprintf('firfilter energy norm failed for M=%d', M));
            end
        end

        function testBiquadFilterEnergyNormMultipleLengths(tc)
            g = biquadfilter(0.25, 0.04, 'energy');
            for L = tc.lengths
                H      = g.H(L);
                energy = sum(abs(H).^2) / L;
                tc.verifyEqual(energy, 1, 'AbsTol', tc.tol_energy, ...
                    sprintf('biquadfilter energy norm failed at L=%d', L));
            end
        end

    end

    % ── Peak normalization across lengths ─────────────────────────────────────
    methods (Test)

        function testBlFilterPeakNormMultipleLengths(tc)
            g = blfilter('hann', 0.1, 0.3, 'peak');
            for L = tc.lengths
                H = comp_transferfunction(g, L);
                tc.verifyEqual(max(abs(H)), 1, 'AbsTol', tc.tol_peak, ...
                    sprintf('blfilter peak norm failed at L=%d', L));
            end
        end

        function testFreqFilterPeakNormMultipleLengths(tc)
            g = freqfilter('gauss', 0.05, 0.3, 'peak');
            for L = tc.lengths
                H = comp_transferfunction(g, L);
                tc.verifyEqual(max(abs(H)), 1, 'AbsTol', tc.tol_peak, ...
                    sprintf('freqfilter peak norm failed at L=%d', L));
            end
        end

        function testBiquadFilterPeakNormMultipleLengths(tc)
            g = biquadfilter(0.25, 0.04, 'peak');
            for L = tc.lengths
                H = g.H(L);
                tc.verifyEqual(max(abs(H)), 1, 'AbsTol', tc.tol_peak, ...
                    sprintf('biquadfilter peak norm failed at L=%d', L));
            end
        end

    end

    % ── Scal linearity across constructors ────────────────────────────────────
    methods (Test)

        function testBlFilterScalLinearity(tc)
            rng(42);
            L = 512;
            for trial = 1:10
                s  = 0.5 + 2*rand();
                g1 = blfilter('hann', 0.1, 0.3);
                g2 = blfilter('hann', 0.1, 0.3, 'scal', s);
                H1 = comp_transferfunction(g1, L);
                H2 = comp_transferfunction(g2, L);
                tc.verifyEqual(H2, s*H1, 'AbsTol', tc.tol_peak * norm(H1), ...
                    sprintf('blfilter scal linearity failed for s=%.3f', s));
            end
        end

        function testFreqFilterScalLinearity(tc)
            rng(42);
            L = 512;
            for trial = 1:10
                s  = 0.5 + 2*rand();
                g1 = freqfilter('gauss', 0.05, 0.3);
                g2 = freqfilter('gauss', 0.05, 0.3, 'scal', s);
                H1 = comp_transferfunction(g1, L);
                H2 = comp_transferfunction(g2, L);
                tc.verifyEqual(H2, s*H1, 'AbsTol', tc.tol_peak * norm(H1), ...
                    sprintf('freqfilter scal linearity failed for s=%.3f', s));
            end
        end

        function testFirFilterScalLinearity(tc)
            % firfilter does not have a 'scal' parameter; test that manually
            % scaling h changes the frequency response by the same factor.
            rng(42);
            L = 256;
            for trial = 1:10
                s  = 0.5 + 2*rand();
                g  = firfilter('hann', 32);
                H1 = comp_transferfunction(g, L);
                H2 = s * H1;
                g_scaled_h  = g;
                g_scaled_h.h = s * g.h;
                H2_from_h = comp_transferfunction(g_scaled_h, L);
                tc.verifyEqual(H2_from_h, H2, 'AbsTol', tc.tol_peak * norm(H1), ...
                    sprintf('firfilter manual scal linearity failed for s=%.3f', s));
            end
        end

    end

    % ── Energy vs peak consistency ────────────────────────────────────────────
    methods (Test)

        function testEnergyAndPeakRelationship(tc)
            % For the same filter: H_energy = H_peak * sqrt(L) approximately,
            % since energy norm sets ||H||=sqrt(L) and peak sets max|H|=1.
            % More precisely: H_energy / H_peak = ||H_raw|| / (max|H_raw| * sqrt(L))
            % This ratio should be consistent for the same filter shape.
            L  = 512;
            ge = blfilter('hann', 0.1, 0.3, 'energy');
            gp = blfilter('hann', 0.1, 0.3, 'peak');
            He = comp_transferfunction(ge, L);
            Hp = comp_transferfunction(gp, L);
            % Both are non-zero multiples of the same raw response
            ratio = He ./ Hp;
            ratio_nz = ratio(abs(Hp) > 1e-6 * max(abs(Hp)));
            % All non-zero ratios should be the same constant
            tc.verifyLessThan(std(abs(ratio_nz)) / mean(abs(ratio_nz)), 1e-8, ...
                'energy and peak responses should differ only by a global constant');
        end

    end

end
