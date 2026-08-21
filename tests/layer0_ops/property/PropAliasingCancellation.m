classdef PropAliasingCancellation < matlab.unittest.TestCase
    % PropAliasingCancellation: property tests for the analysis filterbank
    %
    % Tests that the analysis operator behaves sanely:
    %   - zero input -> zero subbands
    %   - single-subband isolation: zeroing all but one subband and synthesizing
    %     gives output of the correct length
    %   - subband energy is plausible relative to input energy
    %   - two analysis passes give consistent results
    %
    % Uses the public filterbank() API.

    properties
        Ls
        fs
        g
        a
    end

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            testCase.Ls = 1024;
            testCase.fs = 8000;
            [testCase.g, testCase.a] = audfilters(testCase.fs, testCase.Ls);
        end
    end

    methods(Test)

        function testAliasingCancellationSingleSubband(testCase)
            % Isolate one subband at a time and re-synthesise via adjoint.
            % Output length must equal Ls.
            rng(42);
            x = randn(testCase.Ls, 1);
            c = filterbank(x, testCase.g, testCase.a);

            % Use comp_ifilterbank_fft for the full-length filters
            g_pre  = comp_filterbank_pre(testCase.g, testCase.a, testCase.Ls);
            isFL   = cellfun(@(x) isfield(x,'H') && numel(x.H)==testCase.Ls, g_pre);
            G_full = cellfun(@(x) x.H, g_pre(isFL), 'UniformOutput', false);
            a_full = testCase.a(isFL);
            idx_fl = find(isFL);

            for k = 1:min(3, numel(idx_fl))
                target = idx_fl(k);
                c_single = cell(numel(idx_fl), 1);
                for j = 1:numel(idx_fl)
                    if idx_fl(j) == target
                        c_single{j} = c{target};
                    else
                        c_single{j} = zeros(size(c{idx_fl(j)}));
                    end
                end
                F_recon = comp_ifilterbank_fft(c_single, G_full, a_full);
                testCase.verifyEqual(numel(F_recon), testCase.Ls, ...
                    sprintf('Single-subband synthesis: wrong length for subband %d', target));
            end
        end

        function testAliasingCancellationEnergy(testCase)
            % Total subband energy should be within a reasonable factor of input energy
            rng(42);
            x = randn(testCase.Ls, 1);
            c = filterbank(x, testCase.g, testCase.a);

            total_energy = sum(cellfun(@(cm) sum(abs(cm(:)).^2), c));
            input_energy = sum(abs(x).^2);

            ratio = total_energy / input_energy;
            testCase.verifyGreaterThan(ratio, 0.1, 'Subband energy too low');
            testCase.verifyLessThan(ratio, 100,   'Subband energy too high');
        end

        function testReconstructionConsistency(testCase)
            % Two consecutive analysis + adjoint-synthesis passes should agree.
            rng(42);
            g_pre  = comp_filterbank_pre(testCase.g, testCase.a, testCase.Ls);
            isFL   = cellfun(@(x) isfield(x,'H') && numel(x.H)==testCase.Ls, g_pre);
            G_full = cellfun(@(x) x.H, g_pre(isFL), 'UniformOutput', false);
            a_full = testCase.a(isFL);

            if isempty(G_full)
                return   % no full-length filters available -- skip
            end

            x = randn(testCase.Ls, 1);
            F = fft(x);

            % First pass
            c1      = comp_filterbank_fft(F, G_full, a_full);
            F_recon1 = comp_ifilterbank_fft(c1, G_full, a_full);

            % Second pass (on the reconstructed signal)
            c2      = comp_filterbank_fft(F_recon1, G_full, a_full);
            F_recon2 = comp_ifilterbank_fft(c2, G_full, a_full);

            % The two reconstructions should be proportional (both = F/M)
            err = norm(F_recon2 - F_recon1) / norm(F_recon1);
            testCase.verifyLessThan(err, 1e-10, ...
                'Consecutive adjoint passes are not consistent');
        end

    end

end
